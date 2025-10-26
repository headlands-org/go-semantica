# Benchmark Documentation Summary

This directory contains performance analysis and baseline measurements for the pure-go-llamas project.

## Files

### BASELINE_PERFORMANCE.md
Comprehensive baseline performance report including:
- System specifications (Apple M1 Pro, ARM64, 8 cores, 16GB RAM)
- Benchmark results for all test scenarios
- CPU profiling analysis with top 5 hotspots
- Memory allocation breakdown
- Performance observations and optimization opportunities

## Quick Reference

### Current Performance (Apple M1 Pro, ARM64)

**End-to-End Inference:**
- Latency: 27.10 ms/operation
- Throughput: 36.9 inferences/second
- Memory: 1.23 MB/operation

**Forward Pass Only (INT8):**
- Latency: 26.15 ms/operation
- Throughput: 38.2 inferences/second
- Memory: 1.23 MB/operation, 245 allocations

**Tokenization:**
- Latency: 81 microseconds/operation
- Throughput: 12,355 texts/second
- Memory: 153 KB/operation, 2,069 allocations

### Top Optimization Targets

1. **Q8_0 Dequantization** (6.76% of runtime)
   - `gguf.ParseQ8_0Block` + `runtime.extractQ8_0Scales`

2. **Worker Pool Synchronization** (80.35% of runtime)
   - Significant goroutine coordination overhead on ARM

3. **Memory Allocations** (1.37 GB total during model load)
   - Buffer pools: 492 MB (35.93%)
   - Model loading: 360.50 MB (26.32%)
   - Tokenizer: 160.09 MB (11.69%)

### Related Files

Profile data available in `/profiles/`:
- `baseline_cpu.prof` - CPU profiling data
- `baseline_mem.prof` - Memory allocation profiling data
- `README.md` - Instructions for viewing profiles

## Reproducing Benchmarks

```bash
# Run all benchmarks
go test -tags=integration -bench=. -benchmem ./internal/runtime ./internal/kernels

# Run with CPU profiling
go test -tags=integration -bench=BenchmarkForward -cpuprofile=cpu.prof ./internal/runtime

# Run with memory profiling
go test -tags=integration -bench=BenchmarkForward -memprofile=mem.prof ./internal/runtime

# View profiles
go tool pprof -text cpu.prof
go tool pprof -http=:8080 cpu.prof
```

## Git Commit Reference

Baseline established at commit: **8c1cdf0** (Achieve 100% tokenization compatibility)

Date: 2025-10-25
