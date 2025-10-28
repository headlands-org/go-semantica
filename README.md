# Pure-Go GGUF Runtime

This repository contains a Go runtime for GGUF embedding models, focused on the Gemma 300M INT8 embedding variant. The code compiles with the standard Go toolchain (no cgo) and can load models either from disk or directly from embedded byte slices.

## Project Layout
- `pkg/ggufembed`: Public API for loading models and generating embeddings.
- `model/`: Helper that embeds `embeddinggemma-300m-Q8_0.gguf` via `go:embed` for self-contained binaries.
- `internal/gguf`: GGUF reader implementation that supports both mmap and in-memory data.
- `internal/runtime`: Execution engine, kernels, and tokenizer integration.
- `cmd/` and `examples/`: CLI utilities and sample programs that exercise the runtime.
- `testdata/` and `scripts/`: Reference artifacts, comparison tools, and benchmark helpers.

## Getting Started
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

To ship without external model files, use the embedded helper:
```go
import "github.com/lth/pure-go-llamas/model"

rt, _ := model.Open() // loads the embedded Gemma weights straight from memory
```

## Performance vs. llama.cpp
Benchmarks below were run on an AMD Ryzen 9 7900 with `embeddinggemma-300m-Q8_0.gguf`. llama.cpp numbers were captured with the reference C++ benchmark in `benchmark_cpp/`.

| Scenario | Metric | pure-go-llamas | llama.cpp | Notes |
|----------|--------|----------------|-----------|-------|
| Idle | Memory usage | 54 MB heap | ~358 MB RSS | Go runtime is more memory-efficient at rest.
| Single short doc (9w) | P50 latency | 49.8 ms | 8.5 ms | ~6× slower; single-document latency remains the main gap.
| Single long doc (49w) | P50 latency | 276.1 ms | 27.6 ms | ~10× slower for long requests.
| Batch 96× short docs | Throughput | 219.8 emb/s | 252.2 emb/s | ~87% of llama.cpp throughput.
| Batch 96× long docs | Throughput | 33.0 emb/s | 31.5 emb/s | Slightly faster in this specific workload.

Interpretation: single-request latency is significantly slower than llama.cpp today, but batch throughput is close, and memory usage is lower in both idle and batch scenarios.

## Status and Limitations
- Only embedding models are supported; text generation is out of scope.
- AVX2 kernels are provided; other SIMD ISAs are not yet implemented.
- Correctness is validated against llama.cpp (cosine similarity ≈ 0.988). Expect small numerical drift when comparing embeddings bit-for-bit.
- The embedded Gemma model inflates binary size by ~300 MB; use it only when the trade-off is acceptable.

## Development
- Build: `go build ./...`
- Tests: `go test ./...`
- Benchmarks: `go test -bench=. ./internal/runtime`

Use `AGENTS.md` for contribution guidelines. For search and edits during development, repository scripts assume `rg` and `apply_patch` workflows.

## License
MIT
