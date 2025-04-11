import torch
import torch.nn as nn

torch.manual_seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Simulated inputs
q = torch.randn(1, 6, 4).to(DEVICE)
k = torch.randn(1, 6, 4).to(DEVICE)
v = torch.randn(1, 6, 4).to(DEVICE)

# We'll simulate 1 query block: tokens 0-1 (can repeat for all blocks)

q_block = q[:, 0:2, :]

# Block size
B, T, D = q.shape
block_size = 2
scale = 1.0 / (D**0.5)

# Initialise outputs and states
acc = torch.zeros_like(q_block).to(DEVICE)      # accumulated value
max_score = None                     # running max for numerical stability
denom = None                         # softmax denominator

# Iterate over key/value blocks (3 blocks for total 6 tokens)

for j in range(0, T, block_size):
    k_block = k[:, j:j+block_size, :] # [1, 2, 4]
    v_block = v[:, j:j+block_size, :] # [1, 2, 4]
    
    # Compute attention scores
    scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale   # [1, 2, 2]
    
    # Compute numerically stable softmax
    local_max = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - local_max)
    local_sum = exp_scores.sum(dim=-1, keepdim=True)
    weighted_v = torch.matmul(exp_scores, v_block)
    
    # First block: just initialize
    if max_score is None:
        max_score = local_max
        denom = local_sum
        acc = weighted_v
    else:
        new_max = torch.maximum(max_score, local_max)
        exp_diff1 = torch.exp(max_score - new_max)
        exp_diff2 = torch.exp(local_max - new_max)

        acc = acc * exp_diff1 + weighted_v * exp_diff2
        denom = denom * exp_diff1 + local_sum * exp_diff2
        max_score = new_max
        

# Final attention output for q_block
flash_output = acc / denom
print("FlashAttention Output (for block 0–1):")
print(flash_output)

# Full attention on q
full_scores = torch.matmul(q, k.transpose(-2, -1))*scale
full_softmax = torch.softmax(full_scores, -1)
full_output = torch.matmul(full_softmax, v)
print(f"full_output: {full_output}")