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

### Linux (Ryzen 9 7900, x86_64)
Benchmarks collected on Ubuntu with `embeddinggemma-300m-Q8_0.gguf`. llama.cpp numbers come from the C++ benchmark in `benchmark_cpp/` compiled against the CPU backend.

| Scenario | Metric | pure-go-llamas | llama.cpp | Notes |
|----------|--------|----------------|-----------|-------|
| Idle | Memory usage | 54 MB heap | ~358 MB RSS | ~0.15× memory; pure Go keeps resident set smaller.
| Single short doc (9w) | P50 latency | 49.8 ms | 8.5 ms | ~6× slower.
| Single long doc (49w) | P50 latency | 276.1 ms | 27.6 ms | ~10× slower.
| Batch 96× short docs | Throughput | 219.8 emb/s | 252.2 emb/s | ~87% of llama.cpp throughput.
| Batch 96× long docs | Throughput | 33.0 emb/s | 31.5 emb/s | Slightly faster (1.05×).

### macOS (M1 Pro, darwin/arm64)
Benchmarks collected on macOS 14 with an Apple M1 Pro.

`llama.cpp (Metal backend)` was installed via Homebrew and exercises the GPU. `llama.cpp (CPU only)` was built from source with `cmake -DGGML_METAL=OFF` to match the Go runtime’s CPU execution path.

| Scenario | Metric | pure-go-llamas | llama.cpp (Metal) | Notes |
|----------|--------|----------------|-------------------|-------|
| Idle | Memory usage | 54 MB heap | 393 MB RSS | ~0.14× memory consumption.
| Single short doc (9w) | P50 latency | 96.4 ms | 9.5 ms | ~10× slower; GPU outpaces CPU-only Go path.
| Single long doc (49w) | P50 latency | 513.2 ms | 11.4 ms | ~45× slower.
| Batch 96× short docs | Throughput | 79.4 emb/s | 1154.9 emb/s | ~7% of Metal throughput; peak memory 104 MB vs 455 MB.
| Batch 96× long docs | Throughput | 12.0 emb/s | 177.8 emb/s | ~7% of Metal throughput; peak memory 152 MB vs 888 MB.

| Scenario | Metric | pure-go-llamas | llama.cpp (CPU only) | Notes |
|----------|--------|----------------|---------------------|-------|
| Idle | Memory usage | 54 MB heap | 373 MB RSS | ~0.14× memory consumption.
| Single short doc (9w) | P50 latency | 96.4 ms | 7.1 ms | ~14× slower even when both are CPU-bound.
| Single long doc (49w) | P50 latency | 513.2 ms | 44.9 ms | ~11× slower.
| Batch 96× short docs | Throughput | 79.4 emb/s | 443.7 emb/s | ~18% of llama.cpp throughput; peak memory 104 MB vs 446 MB.
| Batch 96× long docs | Throughput | 12.0 emb/s | 61.4 emb/s | ~20% of llama.cpp throughput; peak memory 152 MB vs 730 MB.

Reproducing the CPU-only numbers requires building llama.cpp from source: `cmake -B build-cpu -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j`. Point `LLAMA_CPP_PATH` at that checkout when rebuilding `benchmark_cpp`.

Interpretation: on x86_64 CPUs, batch throughput is close to llama.cpp while single-document latency still lags. On Apple Silicon, llama.cpp’s Metal backend is overwhelmingly faster and even its CPU-only build leads the Go runtime, but the pure-Go path still uses noticeably less memory.

## Status and Limitations
- Only embedding models are supported; text generation is out of scope.
- AVX2 kernels are provided; other SIMD ISAs are not yet implemented.
- GPU/Metal acceleration is unavailable; all execution is CPU-bound even on Apple Silicon.
- Correctness is validated against llama.cpp (cosine similarity ≈ 0.988). Expect small numerical drift when comparing embeddings bit-for-bit.
- The embedded Gemma model inflates binary size by ~300 MB; use it only when the trade-off is acceptable.

## Development
- Build: `go build ./...`
- Tests: `go test ./...`
- Benchmarks: `go test -bench=. ./internal/runtime`

Use `AGENTS.md` for contribution guidelines. For search and edits during development, repository scripts assume `rg` and `apply_patch` workflows.

## License
MIT
