# go-semantica: Gemma Embedding & Search Stack

`go-semantica` is a batteries-included, CPU-only embedding stack written in Go. It ships with:

- **EmbeddingGemma 300M** bundled via `go:embed`, so binaries can carry the model with no external files.
- **A pure Go GGUF runtime** tuned for multi-core CPUs (no cgo, no GPU dependencies).
- **Turn-key vector search**: quantized brute-force indexes plus an experimental pure-Go Annoy implementation, and an examples/search app that ties everything together.

If your corpus fits in RAM (think ≤ 10 K rows), you can embed, index, and search entirely in-process: one binary, one CPU, zero services.

## Installation

```bash
go get github.com/headlands-org/go-semantica@v0.0.1
```

Go 1.25 or newer is required. The command pulls in the top-level embedding API and search packages (`search/...`) so you can embed text and build indexes without extra wiring.

To install the CLI utilities (e.g., `gemma-embed`, `cmd/annoy`), run:

```bash
go install github.com/headlands-org/go-semantica/cmd/gemma-embed@v0.0.1
```

## Getting Started
### Embedding in Go
```go
package main

import (
    "context"
    "fmt"
    "math"

    "github.com/headlands-org/go-semantica"
    "github.com/headlands-org/go-semantica/model"
)

func cosine(a, b []float32) float64 {
    var dot, na, nb float64
    for i := range a {
        dot += float64(a[i] * b[i])
        na += float64(a[i] * a[i])
        nb += float64(b[i] * b[i])
    }
    return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func main() {
    // Load the embedded Gemma model straight from your binary.
    rt := model.MustOpen()
    defer rt.Close()

    // Test sentences: two similar, one different.
    sentences := []string{
        "The cat sits on the mat",
        "A feline rests on the rug",
        "Quantum computers use superposition",
    }

    ctx := context.Background()
    // EmbeddingGemma supports different "tasks" and Matryoshka dimensions.
    embs, err := rt.Embed(ctx, sentences, semantica.TaskSemanticSimilarity, semantica.Dimensions512)
    if err != nil {
        panic(err)
    }

    fmt.Printf("cosine(cat vs feline)  = %.3f\n", cosine(embs[0], embs[1]))
    fmt.Printf("cosine(cat vs quantum) = %.3f\n", cosine(embs[0], embs[2]))
}
```

Example output:

```
cosine(cat vs feline)  = 0.883
cosine(cat vs quantum) = 0.602
```
The semantically similar pair (“cat” vs “feline”) scores much higher than the unrelated quantum sentence—exactly what we want from an embedding model.

### Search quick start

```
go run ./examples/search \
  -icons ../one/go/hugeicons/icons.json \
  -dim 256 \
  -quant int8 \
  -query "Save parking spot with location and notes"
```

The example embeds the dataset (batching 128 at a time), builds both Annoy (experimental) and brute-force indexes, writes them to disk (`icons.brute.<dim>.<quant>.idx`), and loads whichever files already exist. Each run prints index sizes, load/build timings, and the top-10 matches—with distance first so you can skim the results—and you can flip `-dim`/`-quant` to explore recall vs. footprint trade-offs.

### Small-data deployment recipe

For a few thousand documents (≈ 8 K icons in the demo):

1. Run the example once during build or release packaging to precompute `*.idx` files for your preferred Matryoshka tier(s) and quantization.
2. Embed the GGUF model (`model.Open`) **and** the brute-force index (`//go:embed`) into your application binary.
3. At startup, call `search/brute.Serializer.Deserialize` on the embedded bytes; the runtime keeps the vectors in a flat slice for efficient mmap or in-memory use.
4. Use `go-semantica` to embed user queries on demand and call `Index.SearchVector`—no external services, queues, or GPUs required.

Annoy is available alongside brute force for larger corpora today; it’s still experimental compared with the quantized brute-force path.

## Quantized brute-force trade-offs

We benchmarked 500 random queries against the Hugeicons dataset (4 495 items). Each row compares a quantized brute index against the 768-dim fp32 baseline: the table lists on-disk size, average search latency, and recall@10 relative to the fp32 index.

> **Note:** The “Hugeicons dataset” is a small corpus of AI-generated titles and descriptions of the icons in the fantastic [Hugeicons](https://hugeicons.com/) project.

| Dim | Quant | Index Size | Avg Search (ms) | Recall vs 768 fp32 | Recall Loss |
|-----|-------|------------|-----------------|--------------------|-------------|
| 768 | fp32  | 13.2 MB    | 0.26            | 100.00 %           | 0.00 %      |
| 768 | int16 | 6.6 MB     | 0.26            | 100.00 %           | 0.00 %      |
| 768 | int8  | 3.3 MB     | 0.26            | 95.08 %            | 4.92 %      |
| 512 | int16 | 4.4 MB     | 0.18            | 88.56 %            | 11.44 %     |
| 512 | int8  | 2.2 MB     | 0.18            | 88.04 %            | 11.96 %     |
| 256 | int16 | 2.2 MB     | 0.10            | 77.58 %            | 22.42 %     |
| 256 | int8  | 1.1 MB     | 0.10            | 77.60 %            | 22.40 %     |
| 128 | int16 | 1.1 MB     | 0.06            | 66.04 %            | 33.96 %     |
| 128 | int8  | 0.57 MB    | 0.06            | 65.80 %            | 34.20 %     |

All measurements were captured with `go run ./examples/search` (building binaries ahead of time and reusing the cached indexes). The brute-force index stores pre-normalised vectors, so search time scales linearly with dimensionality while quantization controls disk/RAM footprint.

> **Note:** On amd64 hosts with AVX2 the dot-product kernels automatically vectorise; other architectures use the portable scalar fallback.
>
> To regenerate the table, run `go run ./cmd/brute-eval -icons ../one/go/hugeicons/icons.json`. The tool embeds the dataset once, evaluates every dimension/quant pair, and prints either Markdown or CSV (see `-h` for flags).

## Runtime performance vs. llama.cpp

### Linux (Ryzen 9 7900, x86_64)
Benchmarks collected on Arch Linux with `embeddinggemma-300m-Q8_0.gguf`. The Go CLI reports wall-clock and combined CPU time (user + sys); the C++ harness now accepts `-threads` (default `2×` logical cores) so llama.cpp can scale beyond its baked-in `GGML_DEFAULT_N_THREADS = 4`. Effective core count is simply `CPU ÷ wall`.

**Single-document latency (P50)**

| Runtime (threads) | Short 9 w | Cores | Long 49 w | Cores | Extra-long ~400 w | Cores |
|-------------------|-----------|-------|-----------|-------|-------------------|-------|
| go-semantica    | 17.3 ms / 60.9 ms | 3.5× | 76.9 ms / 289.2 ms | 3.8× | 1243.6 ms / 3248.5 ms | 2.6× |
| llama.cpp (4)     | 8.1 ms / 31.9 ms | 3.9× | 28.1 ms / 111.6 ms | 4.0× | 281.7 ms / 1123.0 ms | 4.0× |
| llama.cpp (48)    | 38.5 ms / 323.5 ms | 8.4× | 55.0 ms / 526.1 ms | 9.6× | **243.0 ms / 3074.4 ms** | **12.6×** |

**Batch throughput (96×, 20 s runs)**

| Runtime (threads) | Short docs emb/s | CPU / emb | Cores | Long docs emb/s | CPU / emb | Cores |
|-------------------|------------------|-----------|-------|-----------------|-----------|-------|
| go-semantica    | 285.2            | 74.99 ms  | 21.6× | 47.1            | 469.44 ms | 22.5× |
| llama.cpp (4)     | 253.3            | 15.73 ms  | 4.0×  | 31.5            | 125.06 ms | 4.2× |
| llama.cpp (48)    | 266.8            | 46.47 ms  | 12.9× | 42.8            | 354.16 ms | 15.3× |

**Snapshot**

- llama.cpp keeps four cores busy by default; with `-threads` it scales linearly, driving the ~400-token doc to 243 ms wall while consuming ~12–15 cores.  
- go-semantica still sheds parallelism on long sequences (~11 % utilisation), so closing the latency gap now means reducing per-token CPU rather than spawning more goroutines.  
- Our batch path already saturates most of the socket (>21 cores) but each embedding still costs ~5× more CPU than llama.cpp—kernel efficiency remains the bottleneck.  

Idle footprint remains ~54 MB heap for pure Go vs. ~356 MB RSS for llama.cpp.

## Repository layout
- `go-semantica` (root package): public embedding API and prompt helpers.
- `model`: embedded Gemma weights (`MustOpen` for self-contained binaries).
- `search`: shared search interfaces plus quantized brute force and experimental Annoy backends.
- `examples`: runnable demos; `examples/search` ties embedding and search together.
- `cmd`: CLI utilities and benchmarks built on the same packages.

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
- MTEB evaluation: `go build -o bin/gemma-embed ./cmd/gemma-embed` then `pip install mteb` and run `./scripts/run_mteb.py --model model/embeddinggemma-300m-Q8_0.gguf`

Use `AGENTS.md` for contribution guidelines. For search and edits during development, repository scripts assume `rg` and `apply_patch` workflows.

## License
MIT
