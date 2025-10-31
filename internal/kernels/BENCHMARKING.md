# Kernel Benchmarking Guide

This document explains how to run and interpret benchmarks for parallelized kernel operations.

## Quick Start

```bash
# Run all kernel benchmarks with scaling analysis
go test -bench=. -benchtime=100ms -benchmem -cpu=1,2,4,8 ./internal/kernels/

# Run specific operation benchmarks
go test -bench=BenchmarkRoPE ./internal/kernels/
go test -bench=BenchmarkAttention ./internal/kernels/
go test -bench=BenchmarkMatMul ./internal/kernels/
go test -bench=BenchmarkRMSNorm ./internal/kernels/
```

## Benchmark Files

### Core Benchmarks

- **`rope_parallel_bench_test.go`**: RoPE parallelization benchmarks
  - Serial vs parallel comparison
  - Sequence length variations
  - Head count variations
  - CPU scaling analysis

- **`attention_parallel_bench_test.go`**: Attention head parallelization benchmarks
  - Single-query vs multi-query
  - Head count scaling
  - Sequence length scaling
  - Threshold testing

- **`norm_parallel_bench_test.go`**: Normalization benchmarks
  - RMSNorm across dimensions
  - Batched normalization
  - LayerNorm comparison
  - Gemma-specific RMSNorm

- **`qk_norm_bench_test.go`**: Q/K normalization benchmarks
  - Per-head normalization
  - Combined Q+K normalization
  - Head count variations

- **`matmul_parallel_bench_test.go`**: Matrix multiplication benchmarks
  - Tiling effectiveness
  - Batch scaling
  - FP32 vs INT8
  - Dimension scaling

### Existing Benchmarks

- **`attention_bench_test.go`**: Original attention benchmarks
- **`rope_cache_test.go`**: RoPE cache benchmarks
- **`kernels_test.go`**: Basic kernel operation benchmarks

## Benchmark Categories

### 1. RoPE Benchmarks

```bash
# Compare serial vs parallel across workload sizes
go test -bench=BenchmarkRoPEParallelVsSerial -benchtime=100ms -cpu=1,2,4,8 ./internal/kernels/

# Test sequence length variations
go test -bench=BenchmarkRoPESeqLenVariations -benchtime=100ms -cpu=1,2,4 ./internal/kernels/

# Test head count variations
go test -bench=BenchmarkRoPEHeadVariations -benchtime=100ms -cpu=1,2,4 ./internal/kernels/

# CPU scaling analysis
go test -bench=BenchmarkRoPEScaling -benchtime=100ms -cpu=1,2,4,8,16 ./internal/kernels/
```

**Key Metrics:**
- `work` metric shows totalWork (seqLen × nHeads)
- Compare ns/op between Serial and Parallel
- Look for speedup at different CPU counts

### 2. Attention Benchmarks

```bash
# Head parallelization across configurations
go test -bench=BenchmarkAttentionHeadParallelization -benchtime=100ms -cpu=1,2,4,8 ./internal/kernels/

# Head count scaling
go test -bench=BenchmarkAttentionParallelHeadCounts -benchtime=100ms -cpu=1,2,4 ./internal/kernels/

# Sequence length scaling
go test -bench=BenchmarkAttentionSeqLengths -benchtime=100ms -cpu=1,2,4 ./internal/kernels/

# Threshold testing
go test -bench=BenchmarkAttentionThresholds -benchtime=100ms -cpu=1,2,4 ./internal/kernels/

# With/without mask comparison
go test -bench=BenchmarkAttentionWithMask -benchtime=100ms -cpu=1,2,4 ./internal/kernels/
```

**Key Metrics:**
- `work` metric shows total computation
- Single-query vs multi-query behavior
- Speedup should be >1.0x for effective parallelization

### 3. Normalization Benchmarks

```bash
# RMSNorm across dimensions
go test -bench=BenchmarkRMSNormSizes -benchtime=100ms ./internal/kernels/

# Batched normalization
go test -bench=BenchmarkRMSNormBatchSizes -benchtime=100ms ./internal/kernels/

# Sequential batch processing
go test -bench=BenchmarkRMSNormSequentialBatch -benchtime=100ms ./internal/kernels/

# Q/K normalization
go test -bench=BenchmarkQKNormalization -benchtime=100ms ./internal/kernels/

# Combined Q+K
go test -bench=BenchmarkQKNormalizationCombined -benchtime=100ms ./internal/kernels/
```

**Key Metrics:**
- MB/s shows memory bandwidth utilization
- Per-operation time (ns/op)
- Scaling with dimension size

### 4. Matmul Benchmarks

```bash
# Tiling effectiveness
go test -bench=BenchmarkMatMulGGMLTiling -benchtime=100ms ./internal/kernels/

# Batch scaling
go test -bench=BenchmarkMatMulBatchScaling -benchtime=100ms ./internal/kernels/

# Dimension scaling
go test -bench=BenchmarkMatMulDimScaling -benchtime=100ms ./internal/kernels/

# FP32 vs INT8
go test -bench=BenchmarkMatMulFP32VsINT8 -benchtime=100ms ./internal/kernels/

# INT8 performance
go test -bench=BenchmarkMatMulINT8 -benchtime=100ms ./internal/kernels/
```

**Key Metrics:**
- `gops` shows GFLOPS (higher is better)
- MB/s shows memory bandwidth
- Scaling with batch size

## Interpreting Results

### Understanding Metrics

```
BenchmarkExample/Config-4    1000  1234 ns/op  1024 B/op  10 allocs/op  5678 work
                         │       │       │          │          │          │
                         │       │       │          │          │          └─ Custom metric
                         │       │       │          │          └─ Allocations per op
                         │       │       │          └─ Bytes allocated per op
                         │       │       └─ Nanoseconds per operation
                         │       └─ Number of iterations
                         └─ Number of CPUs (-cpu flag)
```

### Speedup Calculation

```
Speedup = Serial Time / Parallel Time

Example:
  Serial:   100 ns/op
  Parallel:  70 ns/op
  Speedup = 100 / 70 = 1.43x (43% faster)
```

### Threshold Analysis

Good parallelization should show:
1. **Below threshold**: Serial ≈ Parallel (no overhead)
2. **At threshold**: Small speedup (1.1-1.2x)
3. **Above threshold**: Clear speedup (1.3x+)

Example from RoPE:
```
seqLen=32 (below):  Serial 13,957ns  Parallel 18,360ns  → 0.76x (overhead)
seqLen=64 (near):   Serial 28,148ns  Parallel 32,990ns  → 0.85x (break-even)
seqLen=128 (above): Serial 59,426ns  Parallel 42,826ns  → 1.39x (good)
```

### CPU Scaling

Ideal scaling: Linear (2x CPUs = 2x speedup)
Real scaling: Sublinear due to:
- Overhead (scheduling, synchronization)
- Memory bandwidth limits
- Cache contention
- Amdahl's Law (serial portions)

Example:
```
-cpu=1:  100 ns/op  (baseline)
-cpu=2:   60 ns/op  (1.67x speedup - good)
-cpu=4:   40 ns/op  (2.50x speedup - excellent)
-cpu=8:   35 ns/op  (2.86x speedup - diminishing returns)
```

## Best Practices

### Running Benchmarks

1. **Warm up the system**: Run a few times before collecting data
2. **Use consistent benchtime**: `-benchtime=100ms` or `-benchtime=1s`
3. **Test multiple CPU counts**: `-cpu=1,2,4,8` to see scaling
4. **Save results**: Redirect to file for comparison

```bash
# Save results for comparison
go test -bench=. -benchtime=100ms -benchmem -cpu=1,2,4,8 ./internal/kernels/ > results_v1.txt

# Compare with previous version
benchstat results_v1.txt results_v2.txt
```

### Analyzing Results

1. **Compare serial vs parallel**: Look for speedup >1.0x
2. **Check threshold behavior**: No regression below threshold
3. **Verify scaling**: Speedup should increase with CPU count
4. **Consider overhead**: Small workloads may show regression

### When to Parallelize

✅ **Good candidates:**
- Large workloads (>1000 operations)
- Long sequences (≥128 tokens)
- Many independent units (≥16 heads)
- Memory-bound operations (not compute-bound)

❌ **Poor candidates:**
- Small workloads (<100 operations)
- Short sequences (<32 tokens)
- Few units (<8 heads)
- Already compute-optimized (SIMD)

## Continuous Benchmarking

### Pre-commit Checks

```bash
# Quick smoke test
go test -bench=. -benchtime=10ms ./internal/kernels/

# Full validation (slower)
go test -bench=. -benchtime=100ms -benchmem ./internal/kernels/
```

### CI Integration

```bash
# Save baseline
go test -bench=. -benchtime=100ms ./internal/kernels/ > baseline.txt

# After changes
go test -bench=. -benchtime=100ms ./internal/kernels/ > current.txt

# Compare
benchstat baseline.txt current.txt
```

### Performance Regression Detection

Use `benchstat` to detect regressions:

```bash
go install golang.org/x/perf/cmd/benchstat@latest
benchstat old.txt new.txt
```

Look for:
- `~`: No significant change
- `+`: Slower (regression)
- `-`: Faster (improvement)

## See Also

- `/BENCHMARK_RESULTS.md`: Comprehensive results and recommendations
- `/internal/kernels/CACHE_OPTIMIZATION.md`: Cache optimization details
- `internal/runtime/benchmark_test.go`: End-to-end benchmarks
