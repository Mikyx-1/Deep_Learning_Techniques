import torch

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Compute FlashAttention-style attention using tiling and numerically stable log-sum-exp reduction
    based on this paper https://arxiv.org/abs/2205.14135.

    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, num_heads, seq_len, dim]
        k (torch.Tensor): Key tensor of shape [batch_size, num_heads, seq_len, dim]
        v (torch.Tensor): Value tensor of shape [batch_size, num_heads, seq_len, dim]

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, num_heads, seq_len, dim]
                      containing the attention results.
    
    Notes:
        - This function simulates a tiled FlashAttention computation in pure PyTorch.
        - It accumulates attention outputs in log-space to avoid numerical overflow/underflow.
        - The implementation loops over row and column tiles.
    """
    block_size = 2  # You can tune this
    batch_size, num_head, seq_len, dim = q.shape
    scale = 1.0 / (dim ** 0.5)

    o = torch.zeros_like(q, device=q.device)
    l = torch.zeros((batch_size, num_head, seq_len), device=q.device)
    m = torch.full((batch_size, num_head, seq_len), float('-inf'), device=q.device)

    for head_id in range(num_head):
        q_head = q[:, head_id, :, :]  # [B, N, d]
        k_head = k[:, head_id, :, :]  # [B, N, d]
        v_head = v[:, head_id, :, :]  # [B, N, d]

        for j in range(0, seq_len, block_size):
            k_block = k_head[:, j:j + block_size, :]  # [B, Bc, d]
            v_block = v_head[:, j:j + block_size, :]  # [B, Bc, d]

            for i in range(0, seq_len, block_size):
                q_block = q_head[:, i:i + block_size, :]  # [B, Br, d]

                attn_scores = torch.einsum('bid,bjd->bij', q_block, k_block)  # [B, Br, Bc]
                attn_scores = attn_scores * scale

                m_prev = m[:, head_id, i:i + block_size]  # [B, Br]
                l_prev = l[:, head_id, i:i + block_size]  # [B, Br]

                m_curr = torch.max(attn_scores, dim=-1).values  # [B, Br]
                m_new = torch.maximum(m_prev, m_curr)

                exp_m_prev = torch.exp(m_prev - m_new)  # [B, Br]
                exp_m_curr = torch.exp(attn_scores - m_curr.unsqueeze(-1))  # [B, Br, Bc]

                l_curr = exp_m_curr.sum(dim=-1)  # [B, Br]
                l_new = exp_m_prev * l_prev + torch.exp(m_curr - m_new) * l_curr  # [B, Br]

                # accumulate output
                weighted_v = torch.einsum('bij,bjd->bid', exp_m_curr, v_block)  # [B, Br, d]
                o_prev = o[:, head_id, i:i + block_size, :]
                o_new = (exp_m_prev.unsqueeze(-1) * l_prev.unsqueeze(-1) * o_prev +
                         torch.exp(m_curr - m_new).unsqueeze(-1) * weighted_v) / l_new.unsqueeze(-1)

                o[:, head_id, i:i + block_size, :] = o_new
                m[:, head_id, i:i + block_size] = m_new
                l[:, head_id, i:i + block_size] = l_new

    return o

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / k.size(-1)**0.5))
    att = torch.softmax(att, dim=-1)
    y = att @ v
    return y

if __name__ == "__main__":
    # Example usage
    batch_size = 16
    num_head = 12
    seq_len = 64
    dim = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    q = torch.randn((batch_size, num_head, seq_len, dim), device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = flash_attention(q, k, v)
    gt = manual_attn(q, k, v)
    print(output.shape)
    print(f"all_close: {torch.allclose(output, gt)}, Max Diff: {torch.abs(output - gt).max()}")
