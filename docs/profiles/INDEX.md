# Profiling Results Index

**Date Generated:** 2025-10-25
**Test Configuration:** Batch=32, Workers=8, Iterations=100

## Documents (Read These First)

1. **QUICK_COMPARISON.md** - Start here! Quick tables and key findings
2. **PROFILE_ANALYSIS_DETAILED.md** - Comprehensive analysis with recommendations
3. **PROFILE_ANALYSIS.md** - Auto-generated summary (less detailed)
4. **README.md** - How to use these profiles and regenerate them

## Profile Files

### Current Configuration (Fine-Grained Parallelism)
- `current/cpu.pprof` - CPU profile (16 KB)
- `current/mem.pprof` - Memory profile (3.9 KB)
- `current/cpu_top30.txt` - Top 30 CPU functions
- `current/mem_alloc_space.txt` - Memory allocation breakdown
- `current/run.log` - Execution log with timing

### Optimized Configuration (Coarse-Grained Parallelism)
- `optimized/cpu.pprof` - CPU profile (15 KB)
- `optimized/mem.pprof` - Memory profile (3.9 KB)
- `optimized/cpu_top30.txt` - Top 30 CPU functions
- `optimized/mem_alloc_space.txt` - Memory allocation breakdown
- `optimized/run.log` - Execution log with timing

## Key Results at a Glance

### Performance: 1.34x Speedup
- Current: 4.338s/iteration (135.6ms/text)
- Optimized: 3.228s/iteration (100.9ms/text)
- **Winner: Optimized (34% faster)**

### Scheduler Overhead: 4% Reduction
- `runtime.findRunnable`: 436s → 418s
- **Winner: Optimized (less scheduler thrash)**

### Goroutine Creations: 9,600x Reduction
- Current: ~7.68 million goroutines
- Optimized: ~800 goroutines
- **Winner: Optimized (massive reduction)**

### Memory Allocations: 1% Reduction
- Current: 7156.5 MB
- Optimized: 7086.5 MB
- **Winner: Optimized (slight improvement)**

## Recommendations

1. **Use coarse-grained parallelism for batch >= 8**
   - Set `DisableMatmulParallel=true`
   - 34% faster for batch=32
   
2. **Further optimization opportunities:**
   - INT8 quantization buffer reuse (57% of allocations)
   - Token caching (26% of allocations)
   - Attention kernel fusion (memory-bound)

## How to Explore

### Quick View
```bash
# Read the summary
cat QUICK_COMPARISON.md

# See top CPU consumers
head -20 current/cpu_top30.txt
head -20 optimized/cpu_top30.txt

# See top memory allocators
head -20 current/mem_alloc_space.txt
head -20 optimized/mem_alloc_space.txt
```

### Interactive Analysis
```bash
# Web UI (most user-friendly)
go tool pprof -http=:8080 current/cpu.pprof
go tool pprof -http=:8081 optimized/cpu.pprof

# Command-line
go tool pprof -top current/cpu.pprof
go tool pprof -list=MatMul current/cpu.pprof
```

### Generate Flame Graphs
```bash
# Install graphviz if needed
brew install graphviz

# Generate SVG flame graphs
go tool pprof -svg current/cpu.pprof > current/flame.svg
go tool pprof -svg optimized/cpu.pprof > optimized/flame.svg
open current/flame.svg
```

---

**Next Steps:** See `PROFILE_ANALYSIS_DETAILED.md` for full analysis and actionable recommendations.
