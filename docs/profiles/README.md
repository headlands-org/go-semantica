# CPU and Memory Profiling Results

This directory contains profiling data comparing two parallelism strategies for batch embedding generation.

## Quick Summary

**Performance Winner: Coarse-Grained (1.34x speedup)**

- **Current (Fine-Grained)**: 4.338s per iteration → 135.6ms per text
- **Optimized (Coarse-Grained)**: 3.228s per iteration → 100.9ms per text
- **Speedup**: 1.34x faster (34% improvement)

## Directory Structure

```
docs/profiles/
├── README.md                          # This file
├── PROFILE_ANALYSIS.md                # Auto-generated summary
├── PROFILE_ANALYSIS_DETAILED.md       # Detailed analysis with recommendations
├── current/                           # Fine-grained parallelism profiles
│   ├── cpu.pprof                      # CPU profile (binary)
│   ├── mem.pprof                      # Memory profile (binary)
│   ├── cpu_top30.txt                  # Top 30 CPU functions
│   ├── mem_top30.txt                  # Top 30 memory allocations
│   ├── mem_alloc_space.txt            # Allocation space breakdown
│   ├── mem_alloc_objects.txt          # Allocation object count
│   └── run.log                        # Execution log
└── optimized/                         # Coarse-grained parallelism profiles
    ├── cpu.pprof                      # CPU profile (binary)
    ├── mem.pprof                      # Memory profile (binary)
    ├── cpu_top30.txt                  # Top 30 CPU functions
    ├── mem_top30.txt                  # Top 30 memory allocations
    ├── mem_alloc_space.txt            # Allocation space breakdown
    ├── mem_alloc_objects.txt          # Allocation object count
    └── run.log                        # Execution log
```

## Configuration Details

### Current (Fine-Grained Parallelism)
- **Strategy**: Parallelism within matrix multiplication operations
- **Flag**: `DisableMatmulParallel=false`
- **Behavior**: Each text processed serially, but each matmul spawns 16 worker goroutines
- **Best for**: Single texts or very small batches (< 4 texts)

### Optimized (Coarse-Grained Parallelism)
- **Strategy**: Parallelism across texts in the batch
- **Flag**: `DisableMatmulParallel=true`
- **Behavior**: Multiple texts processed in parallel by worker pool, each using serial matmul
- **Best for**: Batch processing (>= 8 texts)

### Test Parameters
- **Model**: `model/embeddinggemma-300m-Q8_0.gguf`
- **Batch Size**: 32 texts
- **Workers**: 8 (NumCPU)
- **Iterations**: 100
- **Total Texts**: 3,200 per configuration
- **Hardware**: 8-core CPU

## Key Findings

### 1. Performance (34% faster)
- Iteration time: 4.338s → 3.228s
- Per-text latency: 135.6ms → 100.9ms
- Wall clock time: 434s → 323s

### 2. Scheduler Overhead (4.1% reduction)
- `runtime.findRunnable`: 436.06s → 418.25s
- `pthread_cond_wait`: 432.69s → 413.58s
- Less time searching for work, more time doing work

### 3. Goroutine Creation (9,600x reduction)
- Current: ~7.68 million goroutine creations
- Optimized: ~800 goroutine creations
- Massive reduction in goroutine spawn/destroy overhead

### 4. Memory Allocation (1% reduction)
- Total allocation: 7156.5 MB → 7086.5 MB
- Buffer pool efficiency: 656 MB → 574 MB (-12.5%)
- Per-text allocation: 2.24 MB → 2.21 MB

### 5. Computational Efficiency
- SIMD overhead: 11.79% → 9.50% (more efficient)
- Matmul overhead: 8.19% → 6.81% (more efficient)
- Attention work: 1.02% → 1.87% (more time on actual compute)

## How to Analyze

### Interactive CPU Profile Exploration

```bash
# Web UI (best for exploration)
go tool pprof -http=:8080 docs/profiles/current/cpu.pprof
go tool pprof -http=:8081 docs/profiles/optimized/cpu.pprof

# Command-line (quick top functions)
go tool pprof -top docs/profiles/current/cpu.pprof
go tool pprof -top docs/profiles/optimized/cpu.pprof

# Focus on specific function
go tool pprof -list=MatMul docs/profiles/current/cpu.pprof
go tool pprof -list=Forward docs/profiles/optimized/cpu.pprof
```

### Memory Profile Analysis

```bash
# Top memory allocations
go tool pprof -top -alloc_space docs/profiles/current/mem.pprof
go tool pprof -top -alloc_space docs/profiles/optimized/mem.pprof

# In-use memory (what's still allocated)
go tool pprof -top -inuse_space docs/profiles/current/mem.pprof

# Web UI for memory
go tool pprof -http=:8080 docs/profiles/current/mem.pprof
```

### Generate Flame Graphs

**Note:** Requires graphviz. Install with `brew install graphviz` on macOS.

```bash
# CPU flame graphs
go tool pprof -svg docs/profiles/current/cpu.pprof > docs/profiles/current/cpu_flame.svg
go tool pprof -svg docs/profiles/optimized/cpu.pprof > docs/profiles/optimized/cpu_flame.svg

# Memory flame graphs (allocation space)
go tool pprof -svg -alloc_space docs/profiles/current/mem.pprof > docs/profiles/current/mem_flame.svg
go tool pprof -svg -alloc_space docs/profiles/optimized/mem.pprof > docs/profiles/optimized/mem_flame.svg

# Open in browser (macOS)
open docs/profiles/current/cpu_flame.svg
open docs/profiles/optimized/cpu_flame.svg
```

### Compare Profiles

```bash
# Compare CPU usage of specific functions
echo "=== Current ==="
go tool pprof -list=processJob docs/profiles/current/cpu.pprof
echo "=== Optimized ==="
go tool pprof -list=processJob docs/profiles/optimized/cpu.pprof

# Compare scheduler overhead
grep -E "(goroutine|procyield|findRunnable)" docs/profiles/current/cpu_top30.txt
grep -E "(goroutine|procyield|findRunnable)" docs/profiles/optimized/cpu_top30.txt
```

## Recommendations

Based on this profiling analysis:

### 1. Use Coarse-Grained for Batch Workloads (batch >= 8)
- 34% faster for batch=32
- Lower scheduler overhead
- Predictable latency

### 2. Use Fine-Grained for Single Texts (batch < 4)
- Maximum single-text throughput
- No worker pool overhead
- Better for latency-critical single requests

### 3. Further Optimization Opportunities
1. **INT8 Quantization** (57% of allocations) - Pre-allocate buffers
2. **Tokenization** (26% of allocations) - Implement caching
3. **Attention Kernels** - Kernel fusion and better memory layout
4. **Worker Pool** - Work stealing for load balancing

See `PROFILE_ANALYSIS_DETAILED.md` for full analysis and recommendations.

## Regenerating Profiles

To re-run profiling with different configurations:

```bash
# Run profiling comparison
./scripts/profile-optimization.sh

# Or manually with custom parameters
./cmd/profile-runtime/profile-runtime \
    -model=model/embeddinggemma-300m-Q8_0.gguf \
    -batch=32 \
    -workers=8 \
    -disable-matmul-parallel=false \
    -iterations=100 \
    -cpuprofile=my_cpu.pprof \
    -memprofile=my_mem.pprof
```

## References

- **Main Analysis**: `PROFILE_ANALYSIS_DETAILED.md`
- **Auto-generated Summary**: `PROFILE_ANALYSIS.md`
- **Profiling Tool**: `cmd/profile-runtime/main.go`
- **Profiling Script**: `scripts/profile-optimization.sh`
- **Go pprof Documentation**: https://go.dev/blog/pprof

---

**Generated:** 2025-10-25
**Tool:** Pure-Go GGUF Runtime Profiling Suite
