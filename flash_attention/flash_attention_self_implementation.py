import torch
import torch.nn as nn

torch.manual_seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simulated inputs
q = torch.randn(1, 6, 4).to(DEVICE)
k = torch.randn(1, 6, 4).to(DEVICE)
v = torch.randn(1, 6, 4).to(DEVICE)

B, N, D = q.shape
block_size = 2
scale = 1.0 / (D**0.5)


O = torch.zeros((B, N, D)).to(DEVICE)
l = torch.zeros((B, N)).to(DEVICE)
m = torch.ones((B, N)).to(DEVICE) * -9999999

q_blocks = []
k_blocks = []
v_blocks = []

for i in range(0, N, block_size):
    q_blocks.append(q[:, i:i+block_size, :])
    k_blocks.append(k[:, i:i+block_size, :])
    v_blocks.append(v[:, i:i+block_size, :])


for j in range(len(k_blocks)):
    # Load K_j, V_j from HBM to on-chip SRAM.
    k_j, v_j = q_blocks[j], v_blocks[j]
    
    for i in range(len(q_blocks)):
        # Load Q_i, O_i, l_i, m_i from HBM to on-chip SRAM.
        Q_i, O_i, l_i, m_i = q_blocks[i], O[:, i:i+block_size, :], l[:, i:i+block_size], m[:, i:i+block_size]
        
        # On chip, compute S_ij = Q_i*(K_j)^T \in R^(B_r x B_c).
        S_ij = torch.matmul(Q_i, k_j.transpose(-2, -1))

        # On chip, compute m̅_ij = rowmax(S_ij) \in R^(B_r), P̅_ij = exp(S_ij - m̅_ij) \in R^(B_r x B_c) (pointwise), l̅_ij = rowsum(P̅_ij) \in R^(B_r).
        m_overline_ij = S_ij.max(dim=-1, keepdim=False).values
        P_overline_ij = torch.exp(S_ij - m_overline_ij.unsqueeze(-1))
        l_overline_ij = P_overline_ij.sum(dim=-1, keepdim=False)
        
        # On chip, compute m^(new)_i = max(m_i, m̅_ij) \in R^(B_r), l^(new)_i = e^(m_i - m^(new)_i)*l_i + e^(m̅_ij - m^(new)_i)l̅_ij \in R^(B_r)
        m_i_new = torch.max(m_i, m_overline_ij)
        l_i_new = torch.exp(m_i - m_i_new)*l_i + torch.exp(m_overline_ij - m_i_new)*l_overline_ij
        
        # O_i ← (diag(l_i^new))^{-1} O_i + exp(m_i - m_i^new) O_i + exp(m̅_ij - m_i^new) P̅_ij V_j


        # ChatGPT
        # Compute weighted_v = P̅_ij * V_j  (shape [B, block_size, D])
        weighted_v = torch.matmul(P_overline_ij, v_j)

        # === LINE 12 FROM THE PAPER ===
        # partial_result = e^(m_i - m_i_new)*[diag(l_i)*O_i] + e^(m̅_ij - m_i_new)*[P̅_ij * V_j]
        # Then multiply by diag(l_i_new)^(-1) which is simply dividing row-wise by l_i_new.

        # 1) Multiply O_i by diag(l_i): each row i is scaled by l_i[i].
        #    O_i has shape [B, block_size, D], so do l_i.unsqueeze(-1) to broadcast across D.
        partial_result = (
            torch.exp(m_i - m_i_new).unsqueeze(-1) * (l_i.unsqueeze(-1) * O_i) +
            torch.exp(m_overline_ij - m_i_new).unsqueeze(-1) * weighted_v
        )
        # partial_result shape: [B, block_size, D]

        # 2) diag(l_i_new)^-1 partial_result => elementwise divide each row by l_i_new
        O_i = partial_result / l_i_new.unsqueeze(-1)  # shape [B, block_size, D]

        # Write back to global memory
        O[:, i:i+block_size, :] = O_i
        m[:, i:i+block_size] = m_i_new
        l[:, i:i+block_size] = l_i_new
print(f"O: {O}")

# print("l\u0305")  # Output: m̅