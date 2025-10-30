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

embedding, err := rt.EmbedSingleInput(ctx, ggufembed.EmbedInput{
    Task:    ggufembed.TaskSearchQuery,
    Content: "Hello, world!",
})

// Supported tasks (per the EmbeddingGemma model card):
//   TaskSearchQuery        -> "task: search result | query: ..."
//   TaskSearchDocument     -> "title: {title|\"none\"} | text: ..."
//   TaskQuestionAnswering  -> "task: question answering | query: ..."
//   TaskFactVerification   -> "task: fact checking | query: ..."
//   TaskClassification     -> "task: classification | query: ..."
//   TaskClustering         -> "task: clustering | query: ..."
//   TaskSemanticSimilarity -> "task: sentence similarity | query: ..."
//   TaskCodeRetrieval      -> "task: code retrieval | query: ..."
//   TaskNone               -> leaves the content unchanged.
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
Benchmarks collected on Arch Linux with `embeddinggemma-300m-Q8_0.gguf`. The Go CLI reports wall-clock and combined CPU time (user + sys); the C++ harness now accepts `-threads` (default `2×` logical cores) so llama.cpp can scale beyond its baked-in `GGML_DEFAULT_N_THREADS = 4`. Effective core count is simply `CPU ÷ wall`.

**Single-document latency (P50)**

| Runtime (threads) | Short 9 w | Cores | Long 49 w | Cores | Extra-long ~400 w | Cores |
|-------------------|-----------|-------|-----------|-------|-------------------|-------|
| pure-go-llamas    | 17.3 ms / 60.9 ms | 3.5× | 76.9 ms / 289.2 ms | 3.8× | 1243.6 ms / 3248.5 ms | 2.6× |
| llama.cpp (4)     | 8.1 ms / 31.9 ms | 3.9× | 28.1 ms / 111.6 ms | 4.0× | 281.7 ms / 1123.0 ms | 4.0× |
| llama.cpp (48)    | 38.5 ms / 323.5 ms | 8.4× | 55.0 ms / 526.1 ms | 9.6× | **243.0 ms / 3074.4 ms** | **12.6×** |

**Batch throughput (96×, 20 s runs)**

| Runtime (threads) | Short docs emb/s | CPU / emb | Cores | Long docs emb/s | CPU / emb | Cores |
|-------------------|------------------|-----------|-------|-----------------|-----------|-------|
| pure-go-llamas    | 285.2            | 74.99 ms  | 21.6× | 47.1            | 469.44 ms | 22.5× |
| llama.cpp (4)     | 253.3            | 15.73 ms  | 4.0×  | 31.5            | 125.06 ms | 4.2× |
| llama.cpp (48)    | 266.8            | 46.47 ms  | 12.9× | 42.8            | 354.16 ms | 15.3× |

**Snapshot**

- llama.cpp keeps four cores busy by default; with `-threads` it scales linearly, driving the ~400-token doc to 243 ms wall while consuming ~12–15 cores.  
- pure-go-llamas still sheds parallelism on long sequences (~11 % utilisation), so closing the latency gap now means reducing per-token CPU rather than spawning more goroutines.  
- Our batch path already saturates most of the socket (>21 cores) but each embedding still costs ~5× more CPU than llama.cpp—kernel efficiency remains the bottleneck.  

Idle footprint remains ~54 MB heap for pure Go vs. ~356 MB RSS for llama.cpp.

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
