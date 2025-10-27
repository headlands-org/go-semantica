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
| **Idle Memory** | Heap Allocated | 54 MB | 358 MB (RSS) | 0.15× |
| **Single Short Doc (9w)** | P50 Latency | 49.8 ms | 8.5 ms | 5.9× slower |
| | P95 Latency | 52.5 ms | 8.9 ms | 5.9× slower |
| **Single Long Doc (49w)** | P50 Latency | 276.1 ms | 27.6 ms | 10.0× slower |
| | P95 Latency | 285.9 ms | 30.0 ms | 9.5× slower |
| **Batch Short Docs (96×)** | Throughput | 219.8 emb/sec | 252.2 emb/sec | 0.87× (87%) |
| | Avg Latency | 4.5 ms/emb | 4.0 ms/emb | 1.1× slower |
| | Peak Memory | 183 MB | 408 MB | 0.45× |
| **Batch Long Docs (96×)** | Throughput | 33.0 emb/sec | 31.5 emb/sec | 1.05× (105%) |
| | Avg Latency | 30.3 ms/emb | 31.8 ms/emb | 0.95× (faster) |
| | Peak Memory | 222 MB | 688 MB | 0.32× |
| **Correctness** | vs llama.cpp | 0.988 cosine similarity ✅ | 1.0 (reference) | - |

**Note**: llama.cpp numbers obtained via new C++ benchmark tool (`benchmark_cpp/`) using llama.cpp libraries directly. See `scripts/compare_benchmarks.sh` for automated comparisons.

### Optimization Details

**Hand-written AVX2 assembly** (`matmulInnerLoopAsm`):
- Signed INT8 multiplication via `VPMOVSXBW` (int8→int16) + `VPMADDWD`
- Vectorized FMA accumulation with `VFMADD231PS` (8 floats at once)
- Scale broadcasting with `VBROADCASTSS` to all YMM lanes
- Register-resident accumulators throughout the loop
- Horizontal reduction only at the end

**Validation approach**: Every optimization is validated against llama.cpp reference embeddings saved in `testdata/reference_embedding_full.txt`. Run `./validate_correctness.sh` to verify cosine similarity ≥ 0.98.

### Performance Analysis

**Single-document latency**: Currently 5.9-10.0× slower than llama.cpp for single document inference. This is the primary area for future optimization.

**Batch throughput**: Achieves competitive performance in batch workloads:
- Short docs (96×): 219.8 emb/sec (87% of llama.cpp's 252.2 emb/sec)
- Long docs (96×): 33.0 emb/sec (105% of llama.cpp's 31.5 emb/sec - actually faster!)

**Memory efficiency**: 54 MB idle heap, 183-222 MB peak for batch workloads. Significantly more memory-efficient than llama.cpp (358 MB idle, 408-688 MB peak).

### Why Use This?

- **Pure Go**: No cgo, cross-compiles easily to any platform
- **Validated correctness**: 0.988 cosine similarity vs llama.cpp
- **Competitive batch performance**: 87-105% of llama.cpp throughput for batch workloads
- **Memory efficient**: Uses 2-3× less memory than llama.cpp (54 MB vs 358 MB idle)
- **Readable code**: Straightforward Go vs complex C++ templates
- **Trade-off**: Prioritizes portability and code clarity over raw single-document speed (5.9-10.0× slower for single docs)

## License

MIT
