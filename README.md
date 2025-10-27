# Pure-Go GGUF Runtime

A pure Go (no cgo) runtime for loading and executing GGUF format embedding models. Targets **Embedding Gemma INT8**.

## Features

- **Pure Go**: No cgo, fully portable
- **SIMD Optimized**: AVX2 (x86-64) and NEON (ARM64) kernels
- **Zero-copy loading**: Memory-mapped file I/O
- **INT8 quantization**: Q8_0 quantized weights
- **Fast**: Competitive with llama.cpp on warm inference

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

## Benchmarks

Tested with `embeddinggemma-300m-Q8_0.gguf` (314MB) on AMD Ryzen 9 7900 (24 cores):

```bash
# Run comprehensive benchmark
./benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive
```

### Performance Comparison vs llama.cpp

| Scenario | pure-go-llamas | llama.cpp | Difference |
|----------|---------------|-----------|------------|
| **Single Short Doc (9w)** | 138ms | 210ms | **1.5× faster** ✨ |
| **Single Long Doc (49w)** | 735ms | 230ms | 3.2× slower |
| **Batch Short (96×)** | 82.9 emb/sec | N/A | - |
| **Batch Long (96×)** | 13.1 emb/sec | N/A | - |
| **Idle Memory** | 54 MB | ~300 MB | **5.6× less memory** |

### Key Insights

**Short documents**: Pure-Go is faster due to lower overhead and optimized coarse-grained parallelism.

**Long documents**: llama.cpp is faster due to more aggressive SIMD optimizations in attention and matmul kernels. We're actively working on closing this gap through:
- Loop unrolling and FMA vectorization (targeting ~600ms)
- SIMD-accelerated attention dot products
- Further accumulation optimizations

**Batch throughput**: Excellent scaling with coarse-grained parallelism (text-level worker pools).

**Memory efficiency**: Zero-copy mmap means we use 5-6× less memory than llama.cpp's loaded-into-RAM approach.

**Why fast?** Zero-copy mmap, AVX2 SIMD kernels, INT8 quantization, minimal allocations, cache-optimized block sizes.

## License

MIT
