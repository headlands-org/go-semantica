# CPU Profiling Analysis: Fine-Grained vs Coarse-Grained Parallelism

**Date:** 2025-10-25 20:05:13

## Executive Summary

This analysis compares two parallelism strategies for batch embedding generation:

1. **Current (Fine-Grained)**: `DisableMatmulParallel=false`
   - Parallelism occurs *within* matrix multiplication operations
   - Each text processed serially, but each matmul uses multiple goroutines

2. **Optimized (Coarse-Grained)**: `DisableMatmulParallel=true`
   - Parallelism occurs *across* texts in the batch
   - Multiple texts processed in parallel, each using serial matmul

## Test Configuration

- **Model**: `model/embeddinggemma-300m-Q8_0.gguf`
- **Batch Size**: 32 texts
- **Workers**: NumCPU (auto-detected)
- **Iterations**: 100
- **Test Data**: Mix of short (5 tokens), medium (15 tokens), and long (40 tokens) texts

## Performance Results

### Timing

| Configuration | Avg Time/Iteration | Relative |
|--------------|-------------------|----------|
| Current (Fine-Grained) | 4.338389375s | 1.00x |
| Optimized (Coarse-Grained) | 3.228042037s | TBD |

### CPU Profile Hotspots

#### Current (Fine-Grained)
```
File: profile-runtime
Type: cpu
Time: 2025-10-25 19:52:31 MDT
Duration: 434.03s, Total samples = 798.92s (184.07%)
Showing nodes accounting for 791.26s, 99.04% of 798.92s total
Dropped 208 nodes (cum <= 3.99s)
Showing top 30 nodes out of 36
      flat  flat%   sum%        cum   cum%
   432.69s 54.16% 54.16%    432.69s 54.16%  runtime.pthread_cond_wait
   151.93s 19.02% 73.18%    151.93s 19.02%  runtime.pthread_cond_signal
    94.19s 11.79% 84.97%     94.19s 11.79%  github.com/lth/pure-go-llamas/internal/kernels.dotProductSDOTAsm
    65.47s  8.19% 93.16%    171.54s 21.47%  github.com/lth/pure-go-llamas/internal/kernels.(*matmulWorkerPool).processJob
    20.11s  2.52% 95.68%     20.11s  2.52%  runtime.memclrNoHeapPointers
    11.87s  1.49% 97.16%    106.06s 13.28%  github.com/lth/pure-go-llamas/internal/kernels.dotProductINT8Asm (inline)
     8.14s  1.02% 98.18%      8.36s  1.05%  github.com/lth/pure-go-llamas/internal/kernels.MultiHeadAttentionWithScale
     4.57s  0.57% 98.75%      4.57s  0.57%  runtime.usleep
     2.11s  0.26% 99.02%     21.91s  2.74%  github.com/lth/pure-go-llamas/internal/kernels.QuantizeSymmetricINT8
     0.04s 0.005% 99.02%     33.91s  4.24%  github.com/lth/pure-go-llamas/internal/runtime.(*Model).ForwardINT8
     0.04s 0.005% 99.03%    436.06s 54.58%  runtime.findRunnable
     0.02s 0.0025% 99.03%     23.69s  2.97%  github.com/lth/pure-go-llamas/internal/runtime.(*Model).runAttentionINT8
```

#### Optimized (Coarse-Grained)
```
File: profile-runtime
Type: cpu
Time: 2025-10-25 19:59:49 MDT
Duration: 323.01s, Total samples = 765.74s (237.06%)
Showing nodes accounting for 761.05s, 99.39% of 765.74s total
Dropped 193 nodes (cum <= 3.83s)
Showing top 30 nodes out of 40
      flat  flat%   sum%        cum   cum%
   413.58s 54.01% 54.01%    413.58s 54.01%  runtime.pthread_cond_wait
   172.50s 22.53% 76.54%    172.50s 22.53%  runtime.pthread_cond_signal
    72.74s  9.50% 86.04%     72.74s  9.50%  github.com/lth/pure-go-llamas/internal/kernels.dotProductSDOTAsm
    52.11s  6.81% 92.84%    133.76s 17.47%  github.com/lth/pure-go-llamas/internal/kernels.(*matmulWorkerPool).processJob
    18.67s  2.44% 95.28%     18.67s  2.44%  runtime.memclrNoHeapPointers
    14.33s  1.87% 97.15%     14.62s  1.91%  github.com/lth/pure-go-llamas/internal/kernels.MultiHeadAttentionWithScale
     8.84s  1.15% 98.31%     81.59s 10.66%  github.com/lth/pure-go-llamas/internal/kernels.dotProductINT8Asm (inline)
     6.43s  0.84% 99.15%      6.43s  0.84%  runtime.usleep
     1.74s  0.23% 99.37%     20.36s  2.66%  github.com/lth/pure-go-llamas/internal/kernels.QuantizeSymmetricINT8
     0.02s 0.0026% 99.38%      4.86s  0.63%  runtime.lock2
     0.02s 0.0026% 99.38%    426.29s 55.67%  runtime.schedule
     0.01s 0.0013% 99.38%     36.55s  4.77%  github.com/lth/pure-go-llamas/internal/runtime.(*Model).ForwardINT8
```

### Memory Allocation

#### Allocation Space

| Configuration | Total Alloc Space |
|--------------|------------------|
| Current |  |
| Optimized |  |

#### Allocation Objects

| Configuration | Total Alloc Objects |
|--------------|---------------------|
| Current |  |
| Optimized |  |

## Analysis

### 1. Goroutine Scheduler Overhead

**Question**: Has goroutine scheduler overhead decreased?

**Current (Fine-Grained)**:
- `runtime.goroutine` overhead: 
- `procyield` overhead: 

**Optimized (Coarse-Grained)**:
- `runtime.goroutine` overhead: 
- `procyield` overhead: 

**Finding**: ⚠️  Review scheduler metrics

### 2. Goroutine Creation

**Question**: Are there fewer goroutines being created?

See detailed analysis in CPU profiles. Look for:
- `runtime.newproc` calls
- `runtime.goexit` calls
- Number of goroutines in flight during profiling

**Current Profile**: `docs/profiles/current/cpu_top30.txt`
**Optimized Profile**: `docs/profiles/optimized/cpu_top30.txt`

### 3. Allocation Rate

**Question**: Has allocation rate decreased?

**Finding**: ⚠️  Review allocation metrics

See:
- `docs/profiles/current/mem_alloc_space.txt` vs `docs/profiles/optimized/mem_alloc_space.txt`
- `docs/profiles/current/mem_alloc_objects.txt` vs `docs/profiles/optimized/mem_alloc_objects.txt`

### 4. New Hotspots

**Question**: What are the new hotspots in the optimized version?

Review the top 30 functions in:
- `docs/profiles/optimized/cpu_top30.txt`

Key areas to examine:
- Is actual computation (matmul, attention) now dominating?
- Has runtime overhead moved down the profile?
- Are there any unexpected new bottlenecks?

### 5. Further Optimization Opportunities

**Question**: Are there further optimization opportunities?

Examine:
1. **Memory allocations**: Can we reduce allocations in hot paths?
2. **Data locality**: Are we cache-friendly?
3. **Algorithmic improvements**: Any redundant computation?
4. **SIMD usage**: Are we maximizing vector instructions?

## Flame Graph Analysis

Visual flame graphs available at:
- **Current CPU**: `docs/profiles/current/cpu_flame.svg`
- **Optimized CPU**: `docs/profiles/optimized/cpu_flame.svg`
- **Current Memory**: `docs/profiles/current/mem_alloc_flame.svg`
- **Optimized Memory**: `docs/profiles/optimized/mem_alloc_flame.svg`

Compare flame graph widths to see relative time spent in different code paths.

## Interactive Analysis Commands

```bash
# View current CPU profile interactively
go tool pprof docs/profiles/current/cpu.pprof

# View optimized CPU profile interactively
go tool pprof docs/profiles/optimized/cpu.pprof

# Compare profiles side-by-side
go tool pprof -http=:8080 docs/profiles/current/cpu.pprof
go tool pprof -http=:8081 docs/profiles/optimized/cpu.pprof

# View memory profiles
go tool pprof docs/profiles/current/mem.pprof
go tool pprof docs/profiles/optimized/mem.pprof

# List top goroutine-related functions
grep -E "(goroutine|procyield|newproc|goexit)" docs/profiles/current/cpu_top30.txt
grep -E "(goroutine|procyield|newproc|goexit)" docs/profiles/optimized/cpu_top30.txt

# List top allocation sites
head -30 docs/profiles/current/mem_alloc_space.txt
head -30 docs/profiles/optimized/mem_alloc_space.txt
```

## Recommendations

Based on this analysis:

1. **If Optimized is Faster**:
   - Document the speedup ratio
   - Update default configuration to use coarse-grained parallelism for batch processing
   - Add guidance on when to use each mode

2. **If Current is Faster**:
   - Investigate why coarse-grained underperforms
   - Check for load balancing issues
   - Consider hybrid approach

3. **Next Steps**:
   - Profile with different batch sizes (8, 16, 64, 128)
   - Test on different hardware (different CPU counts)
   - Measure impact on different text length distributions

## Raw Data

All raw profiles and logs are available in:
- `docs/profiles/current/`
- `docs/profiles/optimized/`
