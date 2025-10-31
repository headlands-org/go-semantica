# Similarity with Embedded Model

This example demonstrates semantic similarity comparison using an embedded model.

Simply import the `model` package to use the embedded GGUF file. No external model file needed!

## Build

```bash
go build ./examples/similarity-embedded
```

That's it! No special flags or setup required.

## Run

```bash
./similarity-embedded
```

The resulting binary is self-contained (~300MB) and includes the model file.

## How It Works

```go
import "github.com/headlands-org/go-semantica/model"

rt, err := model.Open()  // Uses embedded model
defer rt.Close()
```

The `model` package:
- Embeds `model/embeddinggemma-300m-Q8_0.gguf` using `//go:embed`
- Automatically writes it to a temp file at runtime (required for mmap)
- Cleans up the temp file when closed

## Comparison with Regular Similarity Example

| Feature | similarity | similarity-embedded |
|---------|------------|---------------------|
| Binary size | Small (~2MB) | Large (~300MB) |
| Requires external model | Yes | No |
| Import | `go-semantica` | `model` |
| Open call | `go-semantica.Open(path)` | `model.MustOpen()` |
| Use case | Development, flexible | Distribution, deployment |
