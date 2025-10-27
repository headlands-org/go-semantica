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

Tested with embeddinggemma-300m-Q8_0.gguf (314MB):

| Platform | Warm Inference | vs llama.cpp |
|----------|----------------|--------------|
| M1 Pro (ARM64) | 50ms | 7.0x faster |
| Ryzen 9 7900 (x86-64) | 17.5ms | 1.09x (competitive) |

**Why fast?** Zero-copy mmap, SIMD kernels, INT8 quantization, minimal allocations.

## License

MIT
