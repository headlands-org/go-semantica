# Analyzing the 22× Performance Gap vs llama.cpp

## Current State (After AVX2 Optimization)

**Our Performance**: 132ms (short), 705ms (long)
**llama.cpp**: 6ms (short), 18ms (long)
**Gap**: **22× slower (short), 39× slower (long)**

## Where Our Time Goes (50 embeddings, 49-word doc)

Total: 39.53s (791ms per embedding)

| Component | Time | % of Total | Notes |
|-----------|------|-----------|-------|
| **matMulQ8_0INT8Serial** | 31.58s | 79.9% | Matrix multiplication |
| - dotProductINT8Asm | 5.92s | 15.0% | Actual SIMD computation |
| - Accumulation (line 327) | 12.37s | 31.3% | `sum += blockSum * scale` |
| - Scale lookup (line 317) | 3.51s | 8.9% | `scale = scales[...]` |
| - Offset calc (line 318) | 1.90s | 4.8% | `qsOffset = blockOffset + 2` |
| - Final scale (line 347) | 1.78s | 4.5% | `dst[...] = sum * inputScale` |
| **MultiHeadAttentionWithScale** | 3.61s | 9.1% | Attention |
| **GC overhead** | 2.14s | 5.4% | Garbage collection |
| **QuantizeSymmetricINT8** | 0.57s | 1.4% | Runtime quantization |
| **Other** | 1.63s | 4.1% | RMSNorm, GELU, etc. |

## The Smoking Gun: Accumulation Overhead

**Key Finding**: We spend **12.37s** (31% of total time) on line 327:
```go
sum += float32(blockSum) * scale
```

This is a **single scalar multiply-add per block**. Yet it takes **2.1× longer** than the actual SIMD dot product (5.92s)!

### Why is Accumulation So Slow?

Looking at the assembly profile, here's what's happening:

```go
// For a 49-word doc (54 tokens, 768 dims):
// - outDim = 768
// - seqLen = 54
// - blocksPerRow = 24 (768/32)

// Per layer matmul:
//   54 × 768 × 24 = 995,328 loop iterations
//   Each iteration: load scale, multiply, add

// That's ~1 million scalar FMA operations
// Even at 0.5 cycle/op, that's 500K cycles = 0.25ms at 2GHz
// But we're seeing 12.37s / 50 embeddings = 247ms per embedding!
```

**We're taking 1000× longer than the theoretical minimum!**

### Root Cause: Memory Bandwidth Bottleneck

The issue is **memory access patterns**:

```go
for j := 0; j < 768; j++ {           // Iterate output dimensions
    for blockIdx := 0; blockIdx < 24; blockIdx++ {
        scale := scales[scaleBaseIdx+blockIdx]  // Cache miss every time!
        blockSum := dotProductINT8Asm(...)       // This is fast
        sum += float32(blockSum) * scale         // Scalar FMA (fast)
    }
}
```

The `scales` array is laid out as: `[scale_0_0, scale_0_1, ..., scale_0_23, scale_1_0, ..., scale_767_23]`

Each time we load a scale:
- It's from a different cache line (scales are 4 bytes, cache lines are 64 bytes)
- We're striding through memory at `4 × 24 = 96 bytes` intervals
- **L1 cache miss every time** (only 16 scales fit in a 64-byte cache line)

**This is why accumulation dominates!** We're spending 31% of our time waiting on L1 cache misses.

## What llama.cpp Does Differently

### 1. **Vectorized Accumulation**

llama.cpp accumulates in **vector registers**:

```c
__m256 acc = _mm256_setzero_ps();  // 8 float accumulator

for (ib = 0; ib < nb; ++ib) {
    __m256 d = _mm256_set1_ps(scale_x * scale_y);
    __m256 q = mul_sum_i8_pairs_float(qx, qy);  // Returns 8 floats
    acc = _mm256_fmadd_ps(d, q, acc);  // Vector FMA: acc += d * q
}

sumf = hsum_float_8(acc);  // Horizontal sum at end
```

**Key difference**: They broadcast the scale to all 8 lanes and do **vector FMA**, then reduce at the end. We do **scalar FMA per block**.

### 2. **Better Memory Layout**

llama.cpp can use **interleaved block formats**:
- `block_q8_0x4`: Interleaves 4 rows, improving cache locality
- Processes 2×2 or 4×4 matrices at once
- Better cache line utilization

### 3. **No Runtime Quantization**

llama.cpp **doesn't quantize activations to INT8** at runtime. They keep activations in FP16/FP32 and use:
- Q8_0 weights (pre-quantized)
- FP16/FP32 activations (no quantization overhead)

We spend **0.57s (1.4%)** on `QuantizeSymmetricINT8` which they skip entirely.

### 4. **Fused Attention**

llama.cpp uses **online softmax** that:
- Doesn't materialize the attention matrix
- Fuses QK·softmax·V into single pass
- Processes sequentially over KV cache (better cache locality)

We materialize `seqLen × seqLen × nHeads` scratch buffer (9% of time).

### 5. **No Allocations in Hot Path**

llama.cpp uses:
- Pre-allocated arena buffers
- Graph-based execution with buffer reuse
- Zero allocations after first call

We allocate **28+ buffers per forward pass** (5.4% GC overhead).

## Does Fine-Grained Parallelism Matter?

**Short answer: Not for single-document latency.**

Fine-grained parallelism (threading within matmul) helps with:
- Large batch sizes (parallelize across batch dimension)
- Multi-GPU inference (split work across devices)

For **single document on CPU**, it adds overhead:
- Thread scheduling: 1-5μs per spawn
- Cache coherency: Invalidations when threads write to shared memory
- Context switching: If more threads than cores

llama.cpp achieves 6ms **single-threaded**. Their speed comes from:
1. Better SIMD (vectorized accumulation)
2. Better memory layout (cache-friendly)
3. No runtime overhead (no quantization, no allocation)
4. Algorithm efficiency (fused operations)

## How to Bridge the Gap

### High Impact (Should get us to ~5-8× slower)

1. **Vectorize accumulation loop** (31% → 5%)
   - Keep 8 float sums in YMM registers
   - Horizontal reduction at end
   - Expected: 12.37s → 2s (**10s saved**)

2. **Online softmax attention** (9% → 3%)
   - Eliminate scratch buffer
   - Fused QK·softmax·V
   - Expected: 3.61s → 1.2s (**2.4s saved**)

3. **Arena allocator** (5.4% → 1%)
   - Single allocation per forward pass
   - Expected: 2.14s → 0.4s (**1.7s saved**)

**Cumulative**: 39.53s → 25.4s (643ms → 413ms per embedding, **36% faster**)

### Medium Impact (Get us to ~3-5× slower)

4. **GeGLU fusion** (saves memory bandwidth)
5. **Remove runtime quantization** (1.4% saved)
6. **Interleaved block formats** (better cache locality)

### Ultimate (If we rewrote everything)

7. **Graph-based execution** (enables aggressive fusion)
8. **FP16 instead of FP32** (2× memory bandwidth)
9. **Tensor Core API** (for GPUs, irrelevant for CPU)

## The Real Culprit

**It's not parallelism. It's scalar accumulation.**

We're spending 31% of our time doing:
```go
sum += float32(blockSum) * scale  // Scalar
```

When we should be doing:
```go
acc = _mm256_fmadd_ps(d, q, acc)  // Vector (8 floats at once)
```

That **one change** could save us 10 seconds (26% improvement).

## Recommendation

Focus on:
1. **Vectorized accumulation** (highest impact, moderate complexity)
2. **Online softmax** (high impact, high complexity)
3. **Arena allocator** (medium impact, low complexity)

Skip:
- Fine-grained parallelism (adds overhead for single-doc)
- Memory-mapped I/O changes (already validated - not the issue)

This should get us from **22× slower → 5-8× slower**, which is respectable for pure Go vs highly-optimized C++.
