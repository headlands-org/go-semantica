# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pure-Go GGUF Runtime for Embedding Models - A pure Go (no cgo) implementation for loading and executing GGUF format embedding models, specifically targeting Embedding Gemma INT8. The project uses memory-mapped file I/O for zero-copy tensor access and implements CPU-optimized kernels with SIMD acceleration.

## Build Commands

```bash
# Build CLI tools
go build ./cmd/gemma-embed
go build ./cmd/gguf-inspect

# Build examples
go build ./examples/simple
go build ./examples/embeddinggemma-demo
```

## Test Commands

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./internal/kernels ./internal/tokenizer

# Run specific test
go test -v ./internal/runtime -run TestEmbeddingGemma

# Run integration tests (requires model download)
go test -v -tags=integration ./internal/runtime -run TestEmbeddingGemmaVsLlamaCpp

# Run benchmarks
go test -bench=. ./internal/kernels
go test -bench=. ./internal/runtime

# Run benchmarks with CPU profiling
go test -bench=BenchmarkMatMul -cpuprofile=cpu.prof ./internal/kernels
```

## Architecture

### Core Components

1. **GGUF Parser (`internal/gguf/`)** - Memory-mapped GGUF file reader with zero-copy tensor access
   - `reader.go`: Main parser with mmap support via `golang.org/x/exp/mmap`
   - `format.go`: GGUF format definitions (magic, version, dtypes)
   - `tensor.go`: Tensor descriptors and Q8_0 dequantization
   - Handles GGUF v3 format with metadata and tensor info parsing

2. **Tokenizer (`internal/tokenizer/`)** - SentencePiece/Unigram tokenizer implementation
   - Uses BPE (Byte Pair Encoding) algorithm, not Unigram (despite name)
   - Supports Gemma byte fallback tokens (236000-236255 range)
   - Special token handling via metadata, not string matching
   - Normalization pipeline: NFKC → accent removal → lowercase

3. **Kernels (`internal/kernels/`)** - Pure-Go math operations with SIMD
   - `matmul.go`: Matrix multiplication with GGML semantics (weight @ input.T)
   - `attention.go`: Multi-head attention with RoPE and GQA support
   - `norm.go`: RMSNorm and pooling operations
   - `activation.go`: GELU, SiLU activations
   - `simd_amd64.go`: AVX2 SIMD accelerated dot products (8x speedup)
   - `simd_generic.go`: Fallback for non-AVX2 platforms

4. **Runtime (`internal/runtime/`)** - Execution engine for Gemma architecture
   - `model.go`: Model loading, weight management, forward pass
   - Implements Gemma-specific features:
     - Q/K normalization per attention head
     - Post-attention and post-FFN normalization layers
     - GeGLU activation (GELU gate + element-wise multiply)
     - Input embedding scaling by sqrt(embed_dim)
     - Attention scaling by 1/sqrt(head_dim)
   - Supports Grouped Query Attention (GQA) by expanding KV heads

5. **Public API (`pkg/ggufembed/`)** - High-level interface
   - Thread-safe batched inference with worker pools
   - Context-aware cancellation
   - Configurable via functional options pattern

### Data Flow

```
Text → Tokenizer (BPE) → Token IDs → Embedding Layer (scaled) →
Transformer Layers (pre-norm + attention + post-norm + residual →
                   pre-norm + FFN + post-norm + residual) →
Output Norm → Pooling (mean/cls/max) → L2 Normalize → Embedding Vector
```

### GGUF Memory-Mapped Loading

- File is memory-mapped at open time using `golang.org/x/exp/mmap`
- Tensor data accessed via zero-copy byte slices into mmap region
- Q8_0 tensors dequantized on-demand (block format: 2-byte f16 scale + 32x int8)
- Alignment: tensor data starts at 32-byte aligned offset after metadata

### GGML Semantics

Matrix multiplication follows GGML conventions:
- `MatMulGGML(dst, weight, input, batch, inDim, outDim)`
- Equivalent to: `output = input @ weight.T`
- Weight shape: `[outDim, inDim]` stored row-major
- Input shape: `[batch, inDim]`
- Output shape: `[batch, outDim]`

This differs from standard `C = A @ B` notation - pay attention when adding new kernels.

### Gemma Architecture Specifics

When working with Gemma models, note these architectural differences from standard transformers:

1. **Normalization**: Uses RMSNorm instead of LayerNorm
2. **Post-normalization layers**: Attention and FFN blocks have additional normalization after projection
3. **Q/K normalization**: Queries and keys are normalized per-head after projection
4. **Activation**: Uses GeGLU (gate + up projections with GELU) instead of standard FFN
5. **Scaling**: Input embeddings scaled by sqrt(embed_dim), attention by 1/sqrt(head_dim)

### Tensor Naming Conventions

GGUF tensors follow these patterns:
- Token embeddings: `token_embd.weight`
- Layer N attention: `blk.N.attn_{q,k,v,output}.weight`
- Layer N Q/K norms: `blk.N.attn_{q,k}_norm.weight`
- Layer N norms: `blk.N.{attn,ffn}_norm.weight`, `blk.N.post_{attention,ffw}_norm.weight`
- Layer N FFN: `blk.N.ffn_{gate,up,down}.weight`
- Output norm: `output_norm.weight`

### Performance Notes

- SIMD acceleration (AVX2) provides 8x speedup on dot products
- Parallel matmul uses 16 workers (optimized for 24-core CPUs)
- Block sizes: matmul=16 (L1 cache), attention scratch varies by seq_len
- Target: <15ms p50 latency for small texts on 8-core x86

## Common Development Patterns

### Adding New Kernels

1. Add pure-Go implementation in appropriate file (`internal/kernels/`)
2. Add SIMD version in `simd_amd64.go` with build tag `//go:build amd64`
3. Add fallback in `simd_generic.go` with build tag `//go:build !amd64`
4. Write unit tests with golden values
5. Add benchmarks

### Adding New Model Support

1. Update `parseConfig()` in `internal/runtime/model.go` for new metadata keys
2. Update `loadLayer()` for new tensor naming conventions
3. Implement architecture-specific forward pass logic
4. Add integration tests with real model files

### Testing with Real Models

The EmbeddingGemma model is checked into the repository via Git LFS:
```bash
# Model location
ls -lh model/embeddinggemma-300m-Q8_0.gguf

# If model not present, pull from Git LFS
git lfs pull --include="model/embeddinggemma-300m-Q8_0.gguf"
```

Model source: `ggml-org/embeddinggemma-300M-GGUF` on HuggingFace

## CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`):
- Runs on push to master/main and pull requests
- Two jobs: unit tests and integration tests
- Integration tests use Git LFS to pull the checked-in Gemma model
- Validates cosine similarity >= 0.99 vs llama.cpp reference

## Dependencies

Key external dependencies:
- `golang.org/x/exp/mmap` - Memory-mapped file I/O
- `golang.org/x/text/unicode/norm` - Unicode normalization for tokenizer

No cgo required - fully portable across platforms.
