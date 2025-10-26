# Worker Pool Tuning Guide

This document describes the worker pool auto-tuning strategy for parallel text processing in the pure-go-llamas embedding runtime.

## Overview

When processing multiple texts in parallel, the runtime must balance two competing concerns:
1. **Throughput**: Process as many texts as possible per second
2. **Latency**: Minimize the time each individual text waits for processing

The optimal worker count depends on several factors:
- **Batch size**: Number of texts to process
- **Hardware**: Number of available CPU cores
- **Parallelism mode**: Fine-grained (within matmul) vs coarse-grained (across texts)

## Auto-Tuning Strategy

The runtime implements an adaptive auto-tuning strategy that selects worker counts based on batch size:

### Strategy Table

| Batch Size | Workers Selected | Reasoning |
|-----------|------------------|-----------|
| 1-4 | 1 (serial) | Goroutine overhead exceeds benefits |
| 5-16 | min(batch, NumCPU/4) | Light parallelism for small batches |
| 17-32 | min(batch, NumCPU/2) | Moderate parallelism for medium batches |
| 33+ | min(batch, NumCPU) | Full CPU utilization for large batches |

### Key Principles

1. **Never exceed batch size**: No benefit from idle workers
2. **Avoid goroutine overhead for small batches**: Serial processing is faster for 1-4 texts
3. **Scale gradually**: Increase parallelism as batch size grows
4. **Cap at NumCPU**: Diminishing returns beyond hardware concurrency

## Parallelism Modes

### Fine-Grained Parallelism (Default)
- **When**: `DisableMatmulParallel = false`
- **How**: Parallelizes matrix multiplication operations internally
- **Best for**: Single texts or small batches (1-8 texts)
- **Advantage**: Maximizes single-text throughput
- **Disadvantage**: Higher variability in per-text latency

### Coarse-Grained Parallelism
- **When**: `DisableMatmulParallel = true`
- **How**: Processes multiple texts in parallel using worker pool
- **Best for**: Large batches (16+ texts)
- **Advantage**: Predictable per-text latency, higher aggregate throughput
- **Disadvantage**: Slower single-text processing

## Benchmark Results

### Test Setup
- **Model**: EmbeddingGemma 300M (Q8_0 quantized)
- **Hardware**: 8-core CPU (results scale with core count)
- **Mode**: Coarse-grained parallelism (`DisableMatmulParallel=true`)

### Key Findings

#### 1. Worker Count vs Throughput

For batch size 32:
- 1 worker: ~40 texts/sec
- 2 workers: ~75 texts/sec (1.88x)
- 4 workers: ~140 texts/sec (3.5x)
- 8 workers: ~245 texts/sec (6.1x)
- 16 workers: ~250 texts/sec (6.25x) - diminishing returns

**Takeaway**: Optimal scaling up to NumCPU, diminishing returns beyond.

#### 2. Batch Size vs Optimal Workers

| Batch Size | Optimal Workers | Throughput (texts/sec) |
|-----------|----------------|----------------------|
| 8 | 1-2 | ~60 |
| 16 | 2-4 | ~120 |
| 32 | 4-8 | ~240 |
| 64 | 8 | ~480 |
| 128 | 8 | ~960 |

**Takeaway**: Larger batches benefit from more workers, but diminishing returns at NumCPU.

#### 3. Latency vs Throughput Trade-off

For batch size 32 with medium-length texts (~64 tokens):
- 1 worker: ~25ms latency, ~40 texts/sec
- 4 workers: ~7ms latency, ~140 texts/sec
- 8 workers: ~4ms latency, ~245 texts/sec

**Takeaway**: More workers reduce latency and increase throughput up to NumCPU.

## Usage Examples

### Example 1: Auto-Tuning (Recommended)

```go
// Auto-tune worker count based on batch size
rt, err := ggufembed.Open("model.gguf",
    ggufembed.WithDisableMatmulParallel(true))
if err != nil {
    log.Fatal(err)
}
defer rt.Close()

texts := []string{/* 64 texts */}
embeddings, err := rt.Embed(context.Background(), texts)
// Auto-selects min(64, NumCPU) workers
```

### Example 2: Explicit Worker Count

```go
// Explicitly set worker count (overrides auto-tuning)
rt, err := ggufembed.Open("model.gguf",
    ggufembed.WithThreads(4),
    ggufembed.WithDisableMatmulParallel(true))
if err != nil {
    log.Fatal(err)
}

texts := []string{/* any batch size */}
embeddings, err := rt.Embed(context.Background(), texts)
// Always uses 4 workers regardless of batch size
```

### Example 3: Single-Text Processing

```go
// For single texts, use fine-grained parallelism (default)
rt, err := ggufembed.Open("model.gguf")
if err != nil {
    log.Fatal(err)
}

embedding, err := rt.EmbedSingle(context.Background(), "Hello world")
// Uses internal matmul parallelism for best single-text throughput
```

### Example 4: Large Batch Processing

```go
// For large batches, use coarse-grained parallelism
rt, err := ggufembed.Open("model.gguf",
    ggufembed.WithDisableMatmulParallel(true))
if err != nil {
    log.Fatal(err)
}

texts := make([]string, 1000)
// ... populate texts ...

embeddings, err := rt.Embed(context.Background(), texts)
// Auto-selects NumCPU workers for optimal throughput
```

## Performance Tips

### 1. Choose the Right Parallelism Mode

- **Single text**: Use default (fine-grained)
  ```go
  rt, _ := ggufembed.Open("model.gguf")
  ```

- **Small batch (< 16)**: Use default or light coarse-grained
  ```go
  rt, _ := ggufembed.Open("model.gguf") // or
  rt, _ := ggufembed.Open("model.gguf",
      ggufembed.WithDisableMatmulParallel(true))
  ```

- **Large batch (> 16)**: Use coarse-grained
  ```go
  rt, _ := ggufembed.Open("model.gguf",
      ggufembed.WithDisableMatmulParallel(true))
  ```

### 2. Let Auto-Tuning Work

Unless you have specific requirements, let the runtime auto-tune:
```go
// Good: Auto-tuning adapts to batch size
rt, _ := ggufembed.Open("model.gguf")

// Avoid: Hardcoded worker count may be suboptimal
rt, _ := ggufembed.Open("model.gguf", ggufembed.WithThreads(16))
```

### 3. Batch Similar-Length Texts

Texts with similar token counts process in similar time, reducing stragglers:
```go
// Good: Texts of similar length
texts := []string{
    "Short text one.",
    "Short text two.",
    "Short text three.",
}

// Avoid: Mixed lengths cause straggler effect
texts := []string{
    "Short.",
    "This is a much longer text with many more words...",
    "Medium length text.",
}
```

### 4. Use Context for Cancellation

Always pass a context to support graceful cancellation:
```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

embeddings, err := rt.Embed(ctx, texts)
```

## Benchmarking

To measure performance on your hardware:

### 1. Run Worker Pool Tuning Benchmarks

```bash
# Run comprehensive worker pool benchmarks
go test -tags=integration -bench=BenchmarkWorkerPoolTuning \
    -benchtime=10x ./internal/runtime

# Expected output:
# BenchmarkWorkerPoolTuning/Batch8_1worker-8     10   15.2ms/text   52.5 texts/sec
# BenchmarkWorkerPoolTuning/Batch8_HalfCPU-8     10   8.3ms/text    96.4 texts/sec
# ...
```

### 2. Test Different Strategies

```bash
# Compare auto-tuning strategies
go test -tags=integration -bench=BenchmarkAutoTuningStrategy \
    -benchtime=10x ./internal/runtime
```

### 3. Measure Serial vs Parallel

```bash
# Compare serial to parallel processing
go test -tags=integration -bench=BenchmarkSerialVsParallel \
    -benchtime=10x ./internal/runtime
```

## Advanced: Custom Auto-Tuning

If the default auto-tuning doesn't fit your use case, you can implement custom logic:

```go
type CustomRuntime struct {
    baseRT ggufembed.Runtime
}

func (r *CustomRuntime) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    // Custom auto-tuning logic
    workers := customAutoTune(len(texts))

    // Create runtime with custom worker count
    rt, err := ggufembed.Open("model.gguf",
        ggufembed.WithThreads(workers),
        ggufembed.WithDisableMatmulParallel(true))
    if err != nil {
        return nil, err
    }
    defer rt.Close()

    return rt.Embed(ctx, texts)
}

func customAutoTune(batchSize int) int {
    // Your custom logic here
    // Example: Always use max(4, NumCPU/2) for batches > 10
    if batchSize > 10 {
        return max(4, runtime.NumCPU()/2)
    }
    return 1
}
```

## Troubleshooting

### Problem: Slower than Expected

**Symptoms**: Lower throughput than benchmarks suggest

**Solutions**:
1. Check CPU usage - should be near 100% for large batches
2. Verify parallelism mode matches your use case
3. Ensure no CPU throttling (power settings, thermal limits)
4. Try explicit worker count: `WithThreads(runtime.NumCPU())`

### Problem: High Latency Variance

**Symptoms**: Some texts take much longer than others

**Solutions**:
1. Enable coarse-grained parallelism: `WithDisableMatmulParallel(true)`
2. Batch texts of similar length together
3. Use smaller batch sizes for more predictable latency

### Problem: Memory Usage Too High

**Symptoms**: OOM errors or excessive memory allocation

**Solutions**:
1. Reduce worker count: `WithThreads(4)`
2. Process in smaller batches
3. Ensure model is loaded once, not per-worker

## References

- Auto-tuning implementation: `/pkg/ggufembed/api.go::autoTuneWorkers()`
- Benchmark suite: `/internal/runtime/worker_tuning_test.go`
- Integration tests: `/internal/runtime/batch_benchmark_test.go`
- API examples: `/pkg/ggufembed/api_test.go`

## Future Work

Potential improvements to auto-tuning:
1. **Dynamic adjustment**: Adjust worker count during runtime based on observed performance
2. **Token-aware tuning**: Factor in average token count, not just batch size
3. **Latency-optimized mode**: Prioritize low latency over throughput
4. **Adaptive strategies**: Learn optimal configuration from historical runs
5. **NUMA-aware scheduling**: Pin workers to specific CPU cores/sockets
