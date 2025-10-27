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

| Scenario | Metric | Value |
|----------|--------|-------|
| **Idle Memory** | Heap Allocated | 54 MB |
| **Single Short Doc (9w)** | P50 Latency | 51.1 ms |
| | P95 Latency | 54.2 ms |
| **Single Long Doc (49w)** | P50 Latency | 304.3 ms |
| | P95 Latency | 309.8 ms |
| **Batch Short Docs (96×)** | Throughput | 219 emb/sec |
| | Avg Latency | 4.6 ms/emb |
| | Peak Memory | 173 MB |
| **Batch Long Docs (96×)** | Throughput | 31 emb/sec |
| | Avg Latency | 32.7 ms/emb |
| | Peak Memory | 198 MB |
| **Correctness** | vs llama.cpp | 0.988 cosine similarity ✅ |

### Optimization Details

**Hand-written AVX2 assembly** (`matmulInnerLoopAsm`):
- Signed INT8 multiplication via `VPMOVSXBW` (int8→int16) + `VPMADDWD`
- Vectorized FMA accumulation with `VFMADD231PS` (8 floats at once)
- Scale broadcasting with `VBROADCASTSS` to all YMM lanes
- Register-resident accumulators throughout the loop
- Horizontal reduction only at the end

**Validation approach**: Every optimization is validated against llama.cpp reference embeddings saved in `testdata/reference_embedding_full.txt`. Run `./validate_correctness.sh` to verify cosine similarity ≥ 0.98.

### Why Use This?

- **Pure Go**: No cgo, cross-compiles easily to any platform
- **Low memory**: 54 MB idle, ~200 MB peak vs llama.cpp's ~300 MB+ load
- **Good batch throughput**: 219 short texts/sec with efficient parallelism
- **Validated correctness**: 0.988 cosine similarity vs llama.cpp
- **Readable code**: Straightforward Go vs complex C++ templates

## License

MIT
