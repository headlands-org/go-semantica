# Implementation Guide

## Overview

This document provides a technical overview of the Pure-Go GGUF runtime implementation for embedding models, specifically targeting Embedding Gemma INT8.

## Architecture

### 1. GGUF Parser (`internal/gguf`)

The GGUF parser provides memory-mapped access to GGUF model files.

**Key Components:**
- `format.go`: Data type definitions and constants
- `reader.go`: File parser with mmap support
- `tensor.go`: Tensor views and Q8_0 dequantization

**Features:**
- Zero-copy memory mapping using `golang.org/x/exp/mmap`
- Support for Q8_0, F32, and other data types
- Unsafe pointer views for direct memory access
- Automatic alignment handling (32-byte boundaries)

**Usage:**
```go
reader, err := gguf.Open("model.gguf")
defer reader.Close()

// Access metadata
val, ok := reader.GetMetadata("embedding.length")

// Access tensors
desc, ok := reader.GetTensor("token_embd.weight")
data, err := reader.GetTensorData(desc.Name)
```

### 2. Tokenizer (`internal/tokenizer`)

SentencePiece/Unigram tokenizer with full normalization support.

**Key Components:**
- `tokenizer.go`: Unigram model with Viterbi segmentation

**Features:**
- NFKC normalization
- Accent removal
- Case normalization
- Special token handling (BOS/EOS/UNK/PAD)
- Byte fallback for unknown characters

**Normalization Pipeline:**
1. NFKC normalization (Unicode compatibility)
2. Accent removal (optional)
3. Lowercase conversion (optional)

**Tokenization Algorithm:**
- Viterbi dynamic programming for optimal segmentation
- Score-based token selection
- SentencePiece prefix (▁) handling

### 3. Kernels (`internal/kernels`)

Pure-Go computational kernels optimized for cache efficiency.

**Implemented Kernels:**

#### Matrix Operations
- `MatMulF32`: F32×F32 → F32 with cache blocking
- `MatMulQ8_0F32`: Q8_0×F32 → F32 with on-the-fly dequantization
- `VecDotF32`: Vector dot product

#### Normalization
- `RMSNorm`: Root Mean Square normalization
- `LayerNorm`: Standard layer normalization

#### Activations
- `SiLU`: Sigmoid Linear Unit (Swish)
- `GELU`: Gaussian Error Linear Unit
- `GELUQuick`: Fast GELU approximation
- `ReLU`: Rectified Linear Unit
- `Softmax`: Softmax with numerical stability

#### Attention
- `MultiHeadAttention`: Multi-head self-attention
- `ApplyRoPE`: Rotary Position Embedding
- `CLSPooling`: Extract [CLS] token
- `MeanPooling`: Mean pooling over sequence
- `MaxPooling`: Max pooling over sequence

**Performance Optimizations:**
- Cache-friendly blocked matrix multiplication (64×64 blocks)
- Loop unrolling where beneficial
- Bounds check elimination via length hoisting
- Contiguous memory access patterns

### 4. Runtime (`internal/runtime`)

Execution engine for Embedding Gemma architecture.

**Model Architecture:**
```
Input Tokens
    ↓
Token Embeddings
    ↓
N × Transformer Layers:
    ├─ RMSNorm
    ├─ Multi-Head Self-Attention (with RoPE)
    ├─ Residual Connection
    ├─ RMSNorm
    ├─ MLP (GeGLU)
    └─ Residual Connection
    ↓
Pooling (CLS/Mean/Max)
    ↓
Output Embedding
```

**Key Features:**
- Automatic model configuration from GGUF metadata
- Lazy weight loading (on-demand F32 conversion)
- Scratch buffer reuse to minimize allocations
- Support for different pooling strategies

**Memory Management:**
- Pre-allocated scratch buffers for intermediate activations
- Buffer reuse across layers
- Minimal heap allocations during inference

### 5. Public API (`pkg/ggufembed`)

High-level API for embedding generation.

**Interface:**
```go
type Runtime interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error)
    EmbedSingle(ctx context.Context, text string) ([]float32, error)
    Close() error
    EmbedDim() int
    MaxSeqLen() int
}
```

**Features:**
- Context-aware execution
- Concurrent batch processing
- Thread pool management
- Configurable options (threads, batch size, verbose)

**Concurrency Model:**
- Semaphore-based thread pool (configurable size)
- Per-batch goroutines
- Safe concurrent access to model weights (read-only)

## Data Types

### Q8_0 Quantization

**Block Structure (34 bytes):**
```
[0-1]   : scale (float16)
[2-33]  : 32 × int8 quantized values
```

**Dequantization:**
```
float32_value = int8_value × scale
```

**Properties:**
- 32 elements per block
- 8 bits per weight
- ~4× compression vs F32
- Minimal accuracy loss for embeddings

## Performance Characteristics

### Memory Usage
- Model file: mmap'd (shared, read-only)
- Token embeddings: ~vocab_size × embed_dim × 4 bytes
- Layer weights: converted to F32 on load
- Scratch buffers: ~max_seq_len × embed_dim × 10 × 4 bytes

### Expected Performance (MVP, pure Go)
- Single embedding: 10-20ms (8-core x86, seq_len=64)
- Throughput: 50-100 embeddings/sec (batched)
- Memory overhead: 50-100MB beyond model size

### Optimization Opportunities (Fast-Follow)
1. **SIMD kernels** (AVX2/AVX-512/NEON)
   - 4-8× speedup for matmul
   - 2-4× speedup for normalization
2. **Packed weight layouts**
   - VNNI-friendly packing for AVX-512
   - Better cache utilization
3. **Kernel fusion**
   - Combine matmul + activation
   - Reduce memory traffic

## Testing Strategy

### Unit Tests
- Kernel correctness (vs reference implementations)
- Tokenizer determinism (golden input/output pairs)
- Numerical accuracy (tolerance checking)

### Integration Tests
- End-to-end embedding generation
- Model loading and resource cleanup
- Error handling and edge cases

### Golden Tests
- Compare outputs with llama.cpp/PyTorch
- Cosine similarity ≥ 0.999
- Per-layer activation dumps

### Benchmarks
- Latency percentiles (p50, p95, p99)
- Throughput (embeddings/sec)
- Memory usage (RSS, working set)

## Build and Deployment

### Build Commands
```bash
# Build CLI tools
go build ./cmd/gemma-embed
go build ./cmd/gguf-inspect

# Run tests
go test ./...

# Run benchmarks
go test -bench=. ./internal/kernels
```

### Binary Size
- Minimal: ~8-12MB (static binary)
- No external dependencies (pure Go)

### Platform Support
- Linux: amd64, arm64
- macOS: amd64, arm64
- Windows: amd64

## Future Enhancements

### Phase 2: ASM Kernels
- [ ] CPU dispatch (cpuid detection)
- [ ] AVX2 matmul kernel
- [ ] AVX-512 VNNI kernel
- [ ] ARM NEON kernel
- [ ] Build tag organization

### Phase 3: Advanced Features
- [ ] Additional quant formats (Q4_0, Q5_0, K-quants)
- [ ] Multi-model support (LLaMA, GPT-J, etc.)
- [ ] Streaming embeddings
- [ ] Model quantization tools

### Phase 4: Production Features
- [ ] gRPC/HTTP server
- [ ] Metrics and monitoring
- [ ] Model caching
- [ ] Multi-GPU support (via external libs)

## References

- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Gemma models: https://ai.google.dev/gemma
- SentencePiece: https://github.com/google/sentencepiece
