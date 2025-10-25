# Usage Guide

## Quick Start

### Installation

```bash
go get github.com/lth/pure-go-llamas
```

### Basic Usage (Library)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
    // Open model
    rt, err := ggufembed.Open("embedding-gemma-int8.gguf")
    if err != nil {
        log.Fatal(err)
    }
    defer rt.Close()

    // Generate embedding
    ctx := context.Background()
    embedding, err := rt.EmbedSingle(ctx, "Hello, world!")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Embedding dimension: %d\n", len(embedding))
    fmt.Printf("First 5 values: %v\n", embedding[:5])
}
```

### Batch Processing

```go
// Process multiple texts efficiently
texts := []string{
    "First document",
    "Second document",
    "Third document",
}

embeddings, err := rt.Embed(ctx, texts)
if err != nil {
    log.Fatal(err)
}

for i, emb := range embeddings {
    fmt.Printf("Text %d: %d dimensions\n", i, len(emb))
}
```

### Advanced Configuration

```go
import "runtime"

rt, err := ggufembed.Open("model.gguf",
    ggufembed.WithThreads(runtime.NumCPU()),  // Use all CPU cores
    ggufembed.WithBatchSize(8),               // Process 8 texts per batch
    ggufembed.WithVerbose(true),              // Enable logging
)
```

## CLI Tools

### gemma-embed

Generate embeddings from command line.

**Basic usage:**
```bash
# From stdin
echo "hello world" | ./gemma-embed -model model.gguf -format json

# From file
./gemma-embed -model model.gguf -input texts.txt -output embeddings.json
```

**Options:**
- `-model PATH`: Path to GGUF model file (required)
- `-input PATH`: Input file, one text per line (default: stdin)
- `-output PATH`: Output file (default: stdout)
- `-format FORMAT`: Output format: json, csv, tsv (default: json)
- `-threads N`: Number of threads (default: NumCPU)
- `-batch N`: Batch size (default: 1)
- `-verbose`: Enable verbose logging
- `-stats`: Show performance statistics

**Examples:**

JSON output:
```bash
./gemma-embed -model model.gguf -input texts.txt -format json -output embeddings.json
```

CSV output:
```bash
./gemma-embed -model model.gguf -input texts.txt -format csv -output embeddings.csv
```

With statistics:
```bash
./gemma-embed -model model.gguf -input texts.txt -stats
```

Performance tuning:
```bash
./gemma-embed -model model.gguf -input texts.txt -threads 16 -batch 32
```

### gguf-inspect

Inspect GGUF model files.

**Usage:**
```bash
./gguf-inspect model.gguf
```

**Output:**
```
GGUF File: model.gguf
Version: 3
Tensor Count: 150
Metadata KV Count: 25

=== Metadata ===
general.architecture          : gemma
general.name                  : embedding-gemma-int8
tokenizer.ggml.model          : llama
...

=== Tensors ===
Total: 150 tensors

token_embd.weight              dtype=Q8_0      shape=[32000, 2048]  size=...
blk.0.attn_norm.weight         dtype=F32       shape=[2048]         size=...
...
```

## API Reference

### Runtime Interface

```go
type Runtime interface {
    // Embed generates embeddings for multiple texts
    Embed(ctx context.Context, texts []string) ([][]float32, error)

    // EmbedSingle generates an embedding for a single text
    EmbedSingle(ctx context.Context, text string) ([]float32, error)

    // Close releases resources
    Close() error

    // EmbedDim returns the embedding dimension
    EmbedDim() int

    // MaxSeqLen returns the maximum sequence length
    MaxSeqLen() int
}
```

### Options

```go
type Options struct {
    NumThreads int  // Number of threads (default: runtime.NumCPU())
    BatchSize  int  // Batch size (default: 1)
    Verbose    bool // Verbose logging (default: false)
}
```

**Option Functions:**
- `WithThreads(n int)`: Set number of threads
- `WithBatchSize(n int)`: Set batch size
- `WithVerbose(v bool)`: Enable/disable verbose logging

## Common Patterns

### Semantic Search

```go
// Build index
documents := []string{
    "The quick brown fox",
    "Machine learning is awesome",
    "Go is a great language",
}

docEmbeddings, err := rt.Embed(ctx, documents)
if err != nil {
    log.Fatal(err)
}

// Query
query := "programming languages"
queryEmb, err := rt.EmbedSingle(ctx, query)
if err != nil {
    log.Fatal(err)
}

// Find most similar
bestIdx := -1
bestSim := float32(-1)
for i, docEmb := range docEmbeddings {
    sim := cosineSimilarity(queryEmb, docEmb)
    if sim > bestSim {
        bestSim = sim
        bestIdx = i
    }
}

fmt.Printf("Most similar: %q (score: %.3f)\n", documents[bestIdx], bestSim)
```

### Cosine Similarity

```go
func cosineSimilarity(a, b []float32) float32 {
    var dot, normA, normB float32
    for i := range a {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return dot / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float32) float32 {
    return float32(math.Sqrt(float64(x)))
}
```

### Parallel Processing with Worker Pool

```go
type Job struct {
    Text   string
    Result []float32
    Err    error
}

func processInParallel(rt ggufembed.Runtime, texts []string, workers int) [][]float32 {
    jobs := make(chan *Job, len(texts))
    results := make([]*Job, len(texts))

    // Create worker pool
    var wg sync.WaitGroup
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                job.Result, job.Err = rt.EmbedSingle(context.Background(), job.Text)
            }
        }()
    }

    // Submit jobs
    for i, text := range texts {
        job := &Job{Text: text}
        results[i] = job
        jobs <- job
    }
    close(jobs)

    // Wait for completion
    wg.Wait()

    // Extract results
    embeddings := make([][]float32, len(texts))
    for i, job := range results {
        if job.Err != nil {
            log.Printf("Error processing %q: %v", job.Text, job.Err)
            continue
        }
        embeddings[i] = job.Result
    }

    return embeddings
}
```

## Performance Tips

1. **Use batching** for multiple texts:
   ```go
   embeddings, _ := rt.Embed(ctx, texts)  // Better
   // vs
   for _, text := range texts {
       rt.EmbedSingle(ctx, text)           // Slower
   }
   ```

2. **Tune thread count** based on your workload:
   - I/O-bound: `NumThreads = NumCPU`
   - CPU-bound: `NumThreads = NumCPU / 2`

3. **Set appropriate batch size**:
   - Small texts: larger batch (16-32)
   - Large texts: smaller batch (4-8)

4. **Reuse Runtime instance**:
   ```go
   rt, _ := ggufembed.Open("model.gguf")
   defer rt.Close()

   // Reuse for many requests
   for request := range requests {
       rt.Embed(ctx, request.Texts)
   }
   ```

5. **Use context for timeouts**:
   ```go
   ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
   defer cancel()

   embeddings, err := rt.Embed(ctx, texts)
   ```

## Troubleshooting

### Out of Memory

**Problem:** Process crashes with OOM

**Solutions:**
- Reduce batch size
- Process texts in smaller chunks
- Use a machine with more RAM
- Ensure sequence length doesn't exceed `MaxSeqLen`

### Slow Performance

**Problem:** Embeddings take too long

**Solutions:**
- Increase thread count
- Use batching
- Check CPU usage (should be near 100%)
- Verify model is memory-mapped (not copied)

### Incorrect Results

**Problem:** Embeddings differ from reference implementation

**Solutions:**
- Verify model file is not corrupted
- Check tokenizer output (should match reference)
- Compare with golden test cases
- Enable verbose logging to debug

### Build Errors

**Problem:** Build fails with dependency errors

**Solutions:**
```bash
go mod tidy
go mod download
go build ./...
```

## Model Compatibility

### Supported Models
- Embedding Gemma (INT8)
- Models with Q8_0 quantization
- SentencePiece/Unigram tokenizer

### Model Requirements
- GGUF format version 3
- Metadata must include:
  - `embedding.length`
  - `block.count`
  - `attention.head_count`
  - `tokenizer.ggml.tokens`
  - `tokenizer.ggml.scores`

### Converting Models

Use `llama.cpp` tools to convert models to GGUF:
```bash
# Convert from PyTorch/Safetensors
python convert.py model.safetensors --outfile model.gguf

# Quantize to INT8
./quantize model.gguf model-q8_0.gguf Q8_0
```

## Next Steps

- See [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details
- Check [examples/](examples/) for more code samples
- Read the [API documentation](https://pkg.go.dev/github.com/lth/pure-go-llamas)
- Report issues at https://github.com/lth/pure-go-llamas/issues
