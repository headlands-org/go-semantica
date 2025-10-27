# Pure-Go GGUF Runtime

A pure Go (no cgo) runtime for loading and executing GGUF format embedding models. Targets **Embedding Gemma INT8**.

## Features

- **Pure Go**: No cgo, fully portable across platforms
- **Optimized**: AVX2 SIMD kernels with vectorized FMA
- **Zero-copy loading**: Memory-mapped file I/O
- **INT8 quantization**: Efficient Q8_0 quantized weights
- **Validated**: 0.988 cosine similarity vs llama.cpp reference embeddings

## Quick Start

```go
import "github.com/lth/pure-go-llamas/pkg/ggufembed"

rt, err := ggufembed.Open("model.gguf")
if err != nil {
    log.Fatal(err)
}
defer rt.Close()

embedding, err := rt.EmbedSingle(ctx, "Hello, world!")
if err != nil {
    log.Fatal(err)
}
```

## CLI

```bash
# Build
go build ./cmd/gemma-embed

# Generate embeddings
echo "hello world" | ./gemma-embed -model model.gguf -format json

# Batch processing
./gemma-embed -model model.gguf -input texts.txt -output embeddings.csv
```

## Build & Test

```bash
# Build tools
go build ./cmd/gemma-embed
go build ./cmd/gguf-inspect

# Run tests
go test ./...

# Run benchmarks
go test -bench=. ./internal/kernels
go test -bench=. ./internal/runtime
```

## Performance

Tested with `embeddinggemma-300m-Q8_0.gguf` (314MB) on AMD Ryzen 9 7900 (12-core, 24 threads):

```bash
# Build and run comprehensive benchmark
go build ./cmd/benchmark
./benchmark -model ./model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive
```

### Results (as of 2025-10-27)

| Scenario | Metric | pure-go-llamas | llama.cpp | Ratio |
|----------|--------|----------------|-----------|-------|
| **Idle Memory** | Heap Allocated | 54 MB | 0 MB (RSS) | - |
| **Single Short Doc (9w)** | P50 Latency | 51.1 ms | 6.5 ms | 7.9× |
| | P95 Latency | 54.2 ms | 7.1 ms | 7.6× |
| **Single Long Doc (49w)** | P50 Latency | 304.3 ms | 15.8 ms | 19.3× |
| | P95 Latency | 309.8 ms | 21.7 ms | 14.3× |
| **Batch Short Docs (96×)** | Throughput | 219 emb/sec | N/A* | - |
| | Avg Latency | 4.6 ms/emb | N/A* | - |
| | Peak Memory | 173 MB | N/A* | - |
| **Batch Long Docs (96×)** | Throughput | 31 emb/sec | N/A* | - |
| | Avg Latency | 32.7 ms/emb | N/A* | - |
| | Peak Memory | 198 MB | N/A* | - |
| **Correctness** | vs llama.cpp | 0.988 cosine similarity ✅ | 1.0 (reference) | - |

*llama.cpp's `llama-embedding` CLI doesn't support batch processing

### Optimization Details

**Hand-written AVX2 assembly** (`matmulInnerLoopAsm`):
- Signed INT8 multiplication via `VPMOVSXBW` (int8→int16) + `VPMADDWD`
- Vectorized FMA accumulation with `VFMADD231PS` (8 floats at once)
- Scale broadcasting with `VBROADCASTSS` to all YMM lanes
- Register-resident accumulators throughout the loop
- Horizontal reduction only at the end

**Validation approach**: Every optimization is validated against llama.cpp reference embeddings saved in `testdata/reference_embedding_full.txt`. Run `./validate_correctness.sh` to verify cosine similarity ≥ 0.98.

### Performance Analysis

**Single-document latency**: Currently 7.9-19.3× slower than llama.cpp for single document inference. This is the primary area for future optimization.

**Batch throughput**: Achieves 219 short docs/sec (4.6ms avg latency) with efficient parallelism. llama.cpp's CLI doesn't expose batch APIs for comparison, though the library supports batching.

**Memory efficiency**: 54 MB idle, ~200 MB peak for batch workloads.

### Why Use This?

- **Pure Go**: No cgo, cross-compiles easily to any platform
- **Validated correctness**: 0.988 cosine similarity vs llama.cpp
- **Efficient batch processing**: Good throughput with worker pool parallelism
- **Readable code**: Straightforward Go vs complex C++ templates
- **Trade-off**: Prioritizes portability and code clarity over raw single-inference speed

## License

MIT
