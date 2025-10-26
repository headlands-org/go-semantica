# Quick Profile Comparison

## Performance Summary

| Metric | Current (Fine-Grained) | Optimized (Coarse-Grained) | Improvement |
|--------|------------------------|----------------------------|-------------|
| **Total Duration** | 434.03s | 323.01s | **25.6% faster** |
| **Avg Time/Iteration** | 4.338s | 3.228s | **25.6% faster** |
| **Avg Time/Text** | 135.6ms | 100.9ms | **25.6% faster** |
| **Throughput** | 7.37 texts/sec | 9.91 texts/sec | **34.4% increase** |

## Top CPU Hotspots (Side-by-Side)

### Scheduler Overhead

| Function | Current | Optimized | Delta |
|----------|---------|-----------|-------|
| `pthread_cond_wait` | 432.69s (54.16%) | 413.58s (54.01%) | -19.11s (-4.4%) |
| `pthread_cond_signal` | 151.93s (19.02%) | 172.50s (22.53%) | +20.57s (+13.5%) |
| `runtime.findRunnable` | 436.06s (54.58%) | 418.25s (54.62%) | -17.81s (-4.1%) |

### Compute Functions

| Function | Current | Optimized | Delta |
|----------|---------|-----------|-------|
| `dotProductSDOTAsm` (SIMD) | 94.19s (11.79%) | 72.74s (9.50%) | -21.45s (-22.8%) |
| `processJob` (matmul) | 65.47s (8.19%) | 52.11s (6.81%) | -13.36s (-20.4%) |
| `MultiHeadAttentionWithScale` | 8.14s (1.02%) | 14.33s (1.87%) | +6.19s (+76.0%) |

**Key Insight:** Optimized version spends less time in SIMD/matmul overhead and more time in actual attention computation.

## Memory Allocation Summary

| Category | Current (MB) | Optimized (MB) | Delta |
|----------|-------------|----------------|-------|
| **Total Allocation** | 7156.5 | 7086.5 | -70 MB (-1.0%) |
| INT8 Quantization | 4046.0 (56.54%) | 4060.2 (57.29%) | +14.2 MB |
| BPE Tokenization | 1861.5 (26.01%) | 1883.0 (26.57%) | +21.5 MB |
| Buffer Pool | 656.0 (9.17%) | 574.0 (8.10%) | -82 MB (-12.5%) |
| Model Loading | 378.0 (5.28%) | 366.0 (5.17%) | -12 MB (-3.2%) |

**Key Insight:** Buffer pool efficiency improved by 12.5%, offsetting slight increases in quantization/tokenization.

## Goroutine Analysis

### Estimated Goroutine Creations

| Configuration | Goroutines/Iteration | Total (100 iterations) | Notes |
|---------------|---------------------|------------------------|-------|
| **Current** | ~76,800 | ~7,680,000 | 16 workers × 150 matmuls × 32 texts |
| **Optimized** | 8 | 800 | 8 persistent workers reused |

**Reduction: 9,600x fewer goroutine creations**

### Scheduler Metrics

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Work search time (`findRunnable`) | 436.06s | 418.25s | -4.1% |
| Wait time (`pthread_cond_wait`) | 432.69s | 413.58s | -4.4% |
| Signal time (`pthread_cond_signal`) | 151.93s | 172.50s | +13.5% (expected) |

**Signal time increased** because worker pool sends completion signals. This is expected and acceptable given the overall speedup.

## Winner: Coarse-Grained Parallelism

**For batch >= 8, coarse-grained parallelism is the clear winner:**
- 1.34x faster (34% speedup)
- 4% less scheduler overhead
- 9,600x fewer goroutine creations
- 1% less memory allocation
- More time in actual computation vs synchronization

**Recommendation:** Use `DisableMatmulParallel=true` for batch workloads.

---

**Full Analysis:** See `PROFILE_ANALYSIS_DETAILED.md`
**Interactive Analysis:** `go tool pprof -http=:8080 docs/profiles/current/cpu.pprof`
