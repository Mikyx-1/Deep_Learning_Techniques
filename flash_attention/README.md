# Minimalistic FlashAttention Implementation

A down-right simple implementation of the attention mechanism from the paper ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135) by Dao et al.

## 1. Introduction

FlashAttention is a revolutionary approach to computing attention that addresses the quadratic memory complexity problem of traditional attention mechanisms. While standard attention requires O(N²) memory for the attention matrix, FlashAttention reduces this to O(N) by using a tiling strategy and online softmax computation.

Tri Dao, author of the FlashAttention paper, observed that modern GPUs are extremely fast at compute but are bottlenecked by reads/writes (memory bandwidth).
FlashAttention trades a bit more computation for significantly fewer memory accesses, and as a result, achieves better speed and memory efficiency compared to traditional attention.

### Key Innovations:
- **Tiling Strategy**: Breaks down the computation into smaller blocks that fit in fast memory (SRAM)
- **Online Softmax**: Computes softmax incrementally without storing the full attention matrix
- **Numerically Stable**: Uses log-sum-exp tricks to prevent overflow/underflow
- **IO-Aware**: Minimizes data movement between slow (HBM) and fast (SRAM) memory

### Memory Complexity Comparison:
- **Traditional Attention**: O(N²) memory for storing attention scores
- **FlashAttention**: O(N) memory by never materializing the full attention matrix

## 2. How It Works

![flash_attn](https://github.com/user-attachments/assets/38a6b37a-8917-41ac-925d-a50222e2ecf8)


FlashAttention works by dividing the Query, Key, and Value matrices into blocks and processing them tile by tile. Here's the step-by-step breakdown:

### Algorithm Overview

```
For each tile of Keys and Values:
    For each tile of Queries:
        1. Load Q_tile, K_tile, V_tile into fast memory
        2. Compute attention scores: S = Q_tile @ K_tile^T * scale
        3. Update running maximum (for numerical stability)
        4. Compute softmax probabilities incrementally
        5. Update output and normalizers
        6. Move to next tile
```

### Detailed Steps

#### Step 1: Initialization
```python
# Initialize output matrix and tracking variables
O = zeros_like(Q)  # Output matrix
l = zeros(batch_size, num_heads, seq_len)  # Row sums (normalizers)
m = full(-inf, batch_size, num_heads, seq_len)  # Row maxima
```

#### Step 2: Tiling Strategy
The algorithm processes the computation in blocks:
- **Block Size**: Typically 32x32 or 64x64 (fits in GPU shared memory)
- **Column Tiles**: Iterate through K,V blocks
- **Row Tiles**: For each K,V block, iterate through Q blocks

#### Step 3: Core Computation Loop
For each tile combination:

**3a. Load Tiles into Fast Memory**
```python
# Load current blocks
Q_tile = Q[i:i+block_size, :]  # Query block
K_tile = K[j:j+block_size, :]  # Key block  
V_tile = V[j:j+block_size, :]  # Value block
```

**3b. Compute Attention Scores**
```python
# Scaled dot-product attention scores
S = Q_tile @ K_tile.T * (1.0 / sqrt(d))
```

**3c. Online Softmax with Numerical Stability**
This is the most crucial part - computing softmax incrementally:

```python
# Previous state
m_prev = m[i:i+block_size]  # Previous row maxima
l_prev = l[i:i+block_size]  # Previous row sums

# Current tile statistics
m_curr = max(S, dim=-1)     # Current row maxima
m_new = max(m_prev, m_curr) # Updated row maxima

# Stable exponentials
exp_prev = exp(m_prev - m_new)  # Rescale previous
exp_curr = exp(S - m_curr.unsqueeze(-1))  # Current probabilities

# Updated normalizers
l_curr = sum(exp_curr, dim=-1)
l_new = exp_prev * l_prev + exp(m_curr - m_new) * l_curr
```

**3d. Update Output**
```python
# Weighted values from current tile
weighted_V = exp_curr @ V_tile

# Combine with previous output
O_new = (exp_prev.unsqueeze(-1) * l_prev.unsqueeze(-1) * O_prev + 
         exp(m_curr - m_new).unsqueeze(-1) * weighted_V) / l_new.unsqueeze(-1)
```

### Mathematical Foundation

The key insight is the **online softmax formula**. For two sets of logits with different maxima:

```
Given: x₁ with max m₁, x₂ with max m₂
New max: m = max(m₁, m₂)

softmax([x₁, x₂]) = [exp(x₁-m), exp(x₂-m)] / (exp(m₁-m)·sum₁ + exp(m₂-m)·sum₂)

Where: sum₁ = Σexp(x₁-m₁), sum₂ = Σexp(x₂-m₂)
```

This allows incremental softmax computation without storing all logits.

### Memory Access Pattern

**Traditional Attention:**
```
HBM: Store Q, K, V (3Nd elements)
HBM: Allocate S = QKᵀ (N² elements) - this is the problem!
HBM → Compute: Load Q, K to compute S (multiple passes, 2Nd + N² reads)
HBM: Store attention scores S (N² writes)
HBM → Compute: Load S for softmax (N² reads)
HBM: Store softmax probabilities P (N² writes)  
HBM → Compute: Load P, V to compute output (N² + Nd reads)
HBM: Store final output O (Nd writes)
Total HBM storage: 3Nd + 2N² (the N² terms dominate for large N)
Total HBM access: ~4Nd + 4N² (quadratic in sequence length)
```

**FlashAttention:**
```
For each tile:
    HBM → SRAM: Load Q_tile (Br·d)
    HBM → SRAM: Load K_tile (Bc·d)
    HBM → SRAM: Load V_tile (Bc·d)
    SRAM: Compute attention (Br·Bc·d operations)
    SRAM → HBM: Update partial O (Br·d), l, m (Br)

Total HBM access: O(Nd)
    (each Q row, K col, V col is loaded once, no full N² materialization)

Peak SRAM: O((Br + 2Bc)·d + Br·Bc) 
    (Q tile + K tile + V tile + partial scores)
```

### CUDA Implementation Details

The CUDA kernel optimizes this further:
- **Shared Memory**: Stores tiles in fast on-chip memory
- **Thread Cooperation**: Each thread handles one row
- **Memory Coalescing**: Optimized memory access patterns
- **Warp-level Primitives**: Efficient reductions and synchronization

## 3. Results: Speed Comparison

### Theoretical Complexity
| Metric | Traditional Attention | FlashAttention |
|--------|----------------------|----------------|
| Time Complexity | O(N²d) | O(N²d) |
| Memory Complexity | O(N² + Nd) | O(Nd) |
| HBM Access | O(N² + Nd) | O(Nd) |

### Benchmark Results (on NVIDIA RTX 3060 12Gb VRAM)

Run this in your terminal (make sure you have at least 1 GPU)

```bash
python benchmark.py
```

```
=== profiling manual attention ===
/home/lehoangviet/Desktop/python_projects/Deep_Learning_Techniques/flash_attention/benchmark.py:31: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
  with torch.autograd.profiler.profile(use_cuda=True) as prof:
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                     aten::matmul         0.14%      91.105us        62.75%      40.890ms      20.445ms      64.000us         0.10%      40.949ms      20.474ms             2  
                                        aten::bmm        44.50%      28.996ms        62.42%      40.669ms      20.335ms      40.786ms        62.55%      40.786ms      20.393ms             2  
                                    aten::softmax         0.03%      21.282us        25.85%      16.844ms      16.844ms      18.000us         0.03%      16.847ms      16.847ms             1  
                                   aten::_softmax         0.05%      33.722us        25.81%      16.819ms      16.819ms      16.829ms        25.81%      16.829ms      16.829ms             1  
                                        aten::mul         0.09%      59.817us        11.28%       7.351ms       7.351ms       7.358ms        11.28%       7.358ms       7.358ms             1  
                                    aten::reshape         0.03%      21.429us         0.08%      52.816us      13.204us      27.000us         0.04%      56.000us      14.000us             4  
                                  aten::transpose         0.06%      38.853us         0.08%      53.245us      53.245us      40.000us         0.06%      54.000us      54.000us             1  
                                     aten::expand         0.04%      26.693us         0.06%      38.264us       9.566us      28.000us         0.04%      41.000us      10.250us             4  
                                 aten::as_strided         0.03%      16.547us         0.03%      16.547us       3.309us      27.000us         0.04%      27.000us       5.400us             5  
                                       aten::view         0.03%      16.956us         0.03%      16.956us       5.652us      20.000us         0.03%      20.000us       6.667us             3  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 65.159ms
Self CUDA time total: 65.208ms

=== profiling minimal flash attention === 
/home/lehoangviet/Desktop/python_projects/Deep_Learning_Techniques/flash_attention/benchmark.py:37: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
  with torch.autograd.profiler.profile(use_cuda=True) as prof:
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               aten::zeros         0.35%      23.816us        48.36%       3.301ms       1.650ms      29.000us         0.87%       3.304ms       1.652ms             2  
               aten::zero_         0.30%      20.162us        45.62%       3.114ms       1.557ms      20.000us         0.60%       3.116ms       1.558ms             2  
               aten::fill_         0.51%      35.097us        45.30%       3.092ms       1.031ms       3.101ms        93.26%       3.101ms       1.034ms             3  
               aten::empty         0.42%      28.574us         2.33%     158.768us      52.923us     163.000us         4.90%     163.000us      54.333us             3  
                aten::full         0.14%       9.662us         0.30%      20.264us      20.264us      12.000us         0.36%      21.000us      21.000us             1  
           cudaEventRecord         0.42%      28.338us         0.42%      28.338us       1.288us       0.000us         0.00%       0.000us       0.000us            22  
          cudaLaunchKernel        46.37%       3.165ms        46.37%       3.165ms     791.234us       0.000us         0.00%       0.000us       0.000us             4  
     cudaStreamIsCapturing         0.02%       1.076us         0.02%       1.076us       1.076us       0.000us         0.00%       0.000us       0.000us             1  
                cudaMalloc         1.89%     129.118us         1.89%     129.118us     129.118us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceGetAttribute         0.01%       0.841us         0.01%       0.841us       0.841us       0.000us         0.00%       0.000us       0.000us             1  
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 6.826ms
Self CUDA time total: 3.325ms

attn values sanity check: True
```

### Why FlashAttention is Faster

1. **Reduced Memory Bandwidth**: Less data movement between HBM and compute units
2. **Better Cache Utilization**: Tiles fit in fast on-chip memory
3. **Eliminated Intermediate Storage**: No need to store N² attention matrix
4. **Improved Parallelization**: Better GPU occupancy with smaller memory footprint

### Use Cases Where FlashAttention Excels

- **Long Sequence Processing**: Documents, books, code repositories
- **High-Resolution Images**: Vision transformers on large images
- **Multi-Modal Models**: Processing long text + image combinations
- **Memory-Constrained Environments**: Deployment on smaller GPUs
- **Batch Processing**: Larger batch sizes with same memory budget


## 4. Express gratitude
This code is mainly based on this [repo][tspeterkim_repo], I just add the guarding mechanism to make sure the code works on various sequence length (the original code only works if sequence lengths are multiple of 32). Therefore, I want to thank him for his implementation, it is really helpful for beginners like me.


## 5. TL;DR

**Problem**: Traditional attention uses O(N²) memory and is slow for long sequences due to storing the full attention matrix.

**Solution**: FlashAttention uses a tiling strategy to compute attention in blocks, never storing the full N² matrix.

**Key Innovations**:
- 🧩 **Tiling**: Break computation into small blocks that fit in fast memory
- 🧮 **Online Softmax**: Compute softmax incrementally using log-sum-exp tricks  
- 📊 **Numerical Stability**: Track running maximum and sum for stable computation
- 🚀 **IO-Awareness**: Minimize slow memory access, maximize fast memory usage

**Benefits**:
- ✅ **Same Results**: Mathematically equivalent to standard attention
- 🔥 **Faster**: 2-8x speedup for long sequences
- 💾 **Memory Efficient**: 50-90% memory reduction
- 📈 **Scalable**: Enables processing of very long sequences previously impossible

**Note**: If you are new to CUDA programming, I have also written a Python-based version of this algorithm so you can easy follow along (it illustrates how it works but it does not utilize the SRAM power on the GPU). Have fun, peace.

**Bottom Line**: FlashAttention makes transformers faster and more memory-efficient without changing the math, enabling AI applications on longer sequences and smaller hardware.

---

*This implementation provides both a educational PyTorch version and an optimized CUDA kernel for production use.*




[tspeterkim_repo]: https://github.com/tspeterkim/flash-attention-minimal
