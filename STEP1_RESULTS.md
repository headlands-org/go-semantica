# Step 1 Results: Vectorized Non-MatMul Operations

**Date:** 2025-10-27
**Goal:** Vectorize attention, element-wise ops, and GELU activation
**Target Speedup:** 3-4×

---

## Performance Improvement Summary

### Single-Document Latency

| Metric | Baseline | After Step 1 | Improvement |
|--------|----------|--------------|-------------|
| **Short doc (9w) P50** | 51.1 ms | 49.2 ms | **1.04× (3.7%)** |
| **Short doc (9w) P95** | 54.2 ms | 52.1 ms | 1.04× (3.9%) |
| **Long doc (49w) P50** | 304.3 ms | 273.1 ms | **1.11× (10.3%)** |
| **Long doc (49w) P95** | 309.8 ms | 283.8 ms | 1.09× (8.4%) |

### Batch Throughput

| Metric | Baseline | After Step 1 | Improvement |
|--------|----------|--------------|-------------|
| **Batch short (96×)** | 219 emb/sec | 228 emb/sec | **1.04× (4.1%)** |
| **Batch long (96×)** | 30.6 emb/sec | 33.8 emb/sec | **1.10× (10.5%)** |
| **Batch short latency** | 4.6 ms/emb | 4.4 ms/emb | 1.05× (4.5%) |
| **Batch long latency** | 32.7 ms/emb | 29.6 ms/emb | 1.10× (9.5%) |

### Memory Usage

| Metric | Baseline | After Step 1 | Change |
|--------|----------|--------------|--------|
| **Idle memory** | 54 MB | 54 MB | No change |
| **Batch short peak** | 173 MB | 161 MB | **-12 MB (-6.9%)** |
| **Batch long peak** | 198 MB | 209 MB | +11 MB (+5.6%) |

---

## Correctness Validation

✅ **Cosine similarity vs llama.cpp:** 0.988009 (≥0.98 threshold)
✅ **All integration tests:** PASS
✅ **Numerical accuracy:** Maintained

---

## Analysis

### Expected vs Actual Performance

**Expected:** 3-4× speedup (based on ~40M FLOPs reduction in non-matmul operations)
**Actual:** 1.04-1.11× speedup (4-11% improvement)

### Why the Discrepancy?

The lower-than-expected improvement reveals an important insight: **matmul still dominates runtime**, even more than the profiling suggested.

**Breakdown:**
- Our profiling showed matmul at 68% of CPU time
- Non-matmul ops (attention QK^T, GELU, vector ops) at ~32%
- But the 32% includes GC overhead (12%) and other operations
- **Actual hot non-matmul compute:** ~15-20% of total time

**Actual impact of Step 1 optimizations:**
- Attention QK^T: 1.49× faster
- Element-wise ops: 6.7× faster
- GELU: 1.94× faster

These are significant improvements, but they affect a smaller portion of total runtime than anticipated.

### Where is the Time Going?

Based on the results, matmul operations (which we already optimized with AVX2 assembly) account for **~80-85% of actual compute time**, not 68% as the flat profile suggested. The cumulative profile was misleading because it included background GC time.

### What This Means for Our Plan

**Good news:**
- Step 1 optimizations are working as designed (individual kernels show expected speedups)
- Correctness maintained (0.988 similarity)
- Memory efficiency improved for batch workloads

**Reality check:**
- To achieve 6-8× total speedup, we need **much more aggressive matmul optimization**
- Steps 2-3 (pre-quantization, register blocking, prefetch) will be critical
- Step 1's 10% improvement on long docs is a solid foundation

---

## Individual Optimization Results

### 1. Attention QK^T Vectorization ✅
- **Micro-benchmark:** 1.49× speedup (518 µs vs 773 µs)
- **Impact on full model:** Modest (attention compute is fast relative to matmul)

### 2. Element-Wise Operations (VecMul, VecAdd, VecScale) ✅
- **Micro-benchmark:** 6.7× average speedup (11× peak)
- **Impact on full model:** Small (called frequently but very fast operations)

### 3. GELU Activation ✅
- **Micro-benchmark:** 1.94× speedup
- **Impact on full model:** Modest (24 calls/inference, but dominated by matmul)
- **Accuracy:** Excellent (max error 0.00022 vs scalar)

---

## Next Steps

Based on these results, **Step 3 (Advanced Matmul Optimizations) is now the highest priority**:

### Critical Path:
1. **Task 3.1: Software Prefetching** - Target 5-10% gain
2. **Task 3.2: Register Blocking (4×4)** - Target 15-20% gain
3. **Task 2.1: Pre-Quantize Weights** - Target 6-7% gain

### Revised Expectations:
- Step 1: ✅ 1.04-1.11× (achieved)
- Step 2: Target 1.06-1.07× (was 1.15×, revised down)
- Step 3: Target 1.25-1.40× (need bigger gains here)
- Step 4: Target 1.10-1.15× (GC reduction)
- Step 5: Target 1.05× (diminishing returns)

**Cumulative target:** 1.04 × 1.07 × 1.35 × 1.12 × 1.05 = **1.77× total**

This would bring us from 51.1ms → **28.9ms** (vs llama.cpp's 6.5ms = 4.4× slower).

To reach our 6-8× goal (6-8ms latency), we need to find additional optimizations beyond the current plan.

---

## Recommendations

1. **Proceed with Steps 2-3 immediately** - Matmul optimization is critical
2. **Consider more aggressive matmul strategies:**
   - Investigate VNNI instructions (even if requires limited inline asm)
   - Explore multi-level cache blocking
   - Test different register blocking patterns (2×4, 4×2, 4×4, 8×2)
3. **Profile again after Step 3** to identify remaining bottlenecks
4. **Set realistic expectations:** 2× total improvement may be achievable, 6-8× is very ambitious without VNNI

---

## Files Modified

### New Files:
- `internal/kernels/attention_scalar_test.go` - Baseline scalar attention for benchmarks
- `STEP1_RESULTS.md` - This file

### Modified Files:
- `internal/kernels/attention.go` - SIMD dot products in attention
- `internal/kernels/activation.go` - Added `GELUQuickSIMD` and `fastExp`
- `internal/kernels/matmul.go` - SIMD dispatchers for VecMul/Add/Scale
- `internal/kernels/simd_amd64.s` - Assembly for vec ops (VecMul, VecAdd, VecScale)
- `internal/kernels/simd_amd64.go` - Go declarations for SIMD functions
- `internal/kernels/simd_generic.go` - Scalar fallbacks
- `internal/kernels/kernels_test.go` - Comprehensive tests and benchmarks
- `internal/runtime/model_int8.go` - FFN uses `GELUQuickSIMD`

### Test Coverage:
- 16 new test cases (attention, vec ops, GELU accuracy)
- 12 new benchmarks (SIMD vs scalar comparison)
- All tests pass with zero allocations

---

## Conclusion

Step 1 successfully implemented all planned optimizations with excellent micro-benchmark results (1.5-11× speedups on individual kernels). However, the end-to-end impact is smaller than expected (4-11%) because matmul dominates more than originally profiled.

**Key lesson:** Optimizing 20% of runtime by 6× only gives 16% total speedup. To achieve our ambitious 6-8× goal, we must focus heavily on the remaining 80% (matmul operations).

The foundation is solid. Steps 2-3 will be critical to reaching our performance targets.
