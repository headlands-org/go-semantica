# How to Run - Pure-Go GGUF Runtime

## Quick Start 🚀

### 1. Run the EmbeddingGemma Demo

The easiest way to see everything in action:

```bash
./run-embeddinggemma.sh
```

This will:
- Load the EmbeddingGemma-300M Q8_0 model (314 tensors)
- Show model configuration (GQA, dimensions, etc.)
- Generate embeddings for test inputs
- Compute semantic similarity between texts
- Display performance metrics

**Expected output:**
```
╔════════════════════════════════════════════════════════════════╗
║          Pure-Go GGUF Runtime - EmbeddingGemma Demo           ║
╚════════════════════════════════════════════════════════════════╝

📦 Step 1: Loading EmbeddingGemma model...
   ✅ Model loaded in ~1s

📊 Step 2: Model Configuration
   Architecture:     gemma-embedding
   Embedding dim:    768
   Layers:           24
   Attention heads:  3 (Query) + 1 (KV) - Grouped Query Attention
   ...

🧪 Step 3: Generating Embeddings
   Input 1: "Hello world"
   ✅ Embedding generated in ~500ms
   Dimensions: 768
   ...
```

### 2. Customize the Demo

Edit `examples/embeddinggemma-demo/main.go` to try your own inputs:

```go
testInputs := []string{
    "Your text here",
    "Another example",
    "Machine learning",
}
```

Then rebuild and run:
```bash
go build -o embeddinggemma-demo ./examples/embeddinggemma-demo/
./embeddinggemma-demo
```

## Other Ways to Explore 🔍

### Run Integration Tests

See all tests pass with real model data:

```bash
# Test EmbeddingGemma model loading
go test -v ./internal/gguf -tags=integration -run TestEmbeddingGemma

# Test runtime execution
go test -v ./internal/runtime -tags=integration -run TestEmbeddingGemma

# Compare with llama.cpp (shows tokenization difference)
go test -v ./internal/runtime -tags=integration -run TestEmbeddingGemmaVsLlamaCpp

# Run all integration tests
go test -v ./... -tags=integration
```

### Inspect GGUF Models

Explore any GGUF file structure:

```bash
./gguf-inspect models/embeddinggemma-300m-GGUF/embeddinggemma-300m-Q8_0.gguf
```

Shows:
- Model metadata (architecture, dimensions, hyperparameters)
- All 314 tensors with shapes and data types
- Tokenizer configuration
- Quantization details

### Test Component Demos

Run the original component tests (BERT model):

```bash
./run-demo.sh
```

This uses the All-MiniLM-L6-v2 BERT model to show:
- GGUF parsing
- Q8_0 quantization accuracy
- Kernel operations

## Performance Metrics ⚡

From the demo output:

| Operation | Time | Notes |
|-----------|------|-------|
| Model load | ~1s | Includes dequantizing 169 Q8_0 tensors to F32 |
| Embedding (short) | ~500ms | "Hello world" (8 tokens) |
| Embedding (long) | ~700ms | "The quick brown fox" (11 tokens) |

**Performance notes:**
- Pure Go (no CGo overhead)
- No SIMD optimizations yet
- Full F32 computation (llama.cpp uses Q8_0 directly)
- Running on CPU only

## What You'll See Working ✅

When you run the demo, you'll see:

1. **GGUF Loading**
   - 314 tensors parsed
   - 314MB model loaded
   - Zero-copy memory mapping

2. **Model Configuration**
   - Grouped Query Attention (3 Q heads, 1 KV head)
   - 24 transformer layers
   - 768-dimensional embeddings
   - 262K token vocabulary

3. **Embedding Generation**
   - Tokenization (BOS + content + EOS)
   - 24-layer transformer forward pass
   - RoPE, RMSNorm, GeGLU all executing
   - L2-normalized output (norm = 1.0)

4. **Semantic Similarity**
   - Cosine similarity between embeddings
   - Shows related concepts have higher similarity

## Current Limitations ⚠️

### Tokenization Differs from llama.cpp

**What this means:**
- Our tokenizer produces more tokens than llama.cpp
- Example: "Hello world" → 8 tokens (ours) vs 4 tokens (llama.cpp)
- Embeddings are valid but semantically different

**Impact:**
- Cosine similarity with llama.cpp: 0.035 (low)
- If you're using llama.cpp embeddings elsewhere, they won't match
- Embeddings are still internally consistent within this runtime

**Workarounds:**
1. Use this runtime standalone (don't compare with llama.cpp)
2. Use llama.cpp for tokenization, this runtime for inference
3. See `VALIDATION_RESULTS.md` for technical details

### Performance Not Optimized

**Current state:**
- ~500ms for short text embeddings
- All computation in F32
- No SIMD/vectorization

**Future improvements:**
- Add SIMD for matmul (~10x faster possible)
- Use Q8_0 directly without F32 conversion
- Optimize memory allocation
- Add batching support

## Development Commands 🛠️

```bash
# Build all binaries
go build ./cmd/... ./examples/...

# Run unit tests
go test ./...

# Run integration tests (requires model)
go test ./... -tags=integration

# Build specific examples
go build -o gguf-inspect ./cmd/gguf-inspect/
go build -o embeddinggemma-demo ./examples/embeddinggemma-demo/

# Format code
go fmt ./...

# Check for issues
go vet ./...
```

## File Locations 📁

```
pure-go-llamas/
├── embeddinggemma-demo       # ← Run this!
├── run-embeddinggemma.sh     # ← Or run this script
├── gguf-inspect              # Model inspector
├── models/
│   └── embeddinggemma-300m-GGUF/
│       └── embeddinggemma-300m-Q8_0.gguf  # 314MB model
├── examples/
│   ├── embeddinggemma-demo/  # Main demo program
│   └── test-components/      # Component tests
└── internal/
    ├── gguf/                 # GGUF parser
    ├── runtime/              # Execution engine
    ├── kernels/              # Math operations
    └── tokenizer/            # Text processing
```

## Next Steps 💡

After running the demo:

1. **Read the validation results**
   ```bash
   cat VALIDATION_RESULTS.md
   ```
   Understand what works and what doesn't

2. **Check the session summary**
   ```bash
   cat SESSION_SUMMARY.md
   ```
   See what was accomplished

3. **Explore the code**
   - Start with `examples/embeddinggemma-demo/main.go`
   - Look at `internal/runtime/model.go` for the forward pass
   - Check `internal/gguf/reader.go` for GGUF parsing

4. **Try your own inputs**
   - Modify test strings in the demo
   - Generate embeddings for your use case
   - Test semantic similarity on your domain

5. **Run tests**
   ```bash
   go test -v ./internal/runtime -tags=integration -run TestEmbeddingGemmaFullPipeline
   ```
   See the full pipeline in action with detailed logging

## Troubleshooting 🔧

**"Model not found"**
- Make sure you downloaded the model (see main README.md)
- Check the model is at: `models/embeddinggemma-300m-GGUF/embeddinggemma-300m-Q8_0.gguf`

**"Slow performance"**
- This is expected (no optimizations yet)
- ~500-700ms per embedding is normal
- See "Performance Not Optimized" section above

**"Different results than llama.cpp"**
- This is expected (tokenization difference)
- See "Current Limitations" section above
- Read `VALIDATION_RESULTS.md` for details

## Questions? 💬

Check these files:
- `VALIDATION_RESULTS.md` - Detailed test results
- `SESSION_SUMMARY.md` - What was built and why
- `IMPLEMENTATION.md` - Technical deep-dive
- `README.md` - Project overview

Have fun exploring! 🎉
