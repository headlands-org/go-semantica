# LlamaModel RAII Wrapper Documentation

## Overview

This document describes the `LlamaModel` C++ RAII wrapper class for llama.cpp model and context management with embedding support.

## Purpose

The `LlamaModel` class provides:
- **Safe resource management**: Automatic cleanup via RAII (Resource Acquisition Is Initialization)
- **Exception-based error handling**: Clear error reporting without manual error code checking
- **Simplified embedding API**: High-level interface for single and batch embedding generation
- **Memory safety**: No memory leaks, even when exceptions occur

## Files

- **Header**: `include/model.h`
- **Implementation**: `src/model.cpp`
- **Test/Example**: `src/main.cpp`

## Class Interface

### Constructor

```cpp
LlamaModel::LlamaModel(
    const std::string& model_path,
    uint32_t n_ctx = 0,          // 0 = use model default
    uint32_t n_batch = 512,
    int32_t n_threads = 0        // 0 = auto-detect
);
```

**Initialization sequence:**
1. Calls `llama_backend_init()` (once, idempotent)
2. Loads model from file using `llama_model_load_from_file()`
3. Gets vocabulary handle with `llama_model_get_vocab()`
4. Creates context with `llama_init_from_model()`
   - Sets `pooling_type = LLAMA_POOLING_TYPE_MEAN` (matching Go implementation)
   - Sets `embeddings = true`
   - Sets `n_seq_max = 64` (supports up to 64 parallel sequences in batch)

**Throws:**
- `ModelError` if model loading fails
- `ModelError` if context initialization fails

### Destructor

```cpp
~LlamaModel();
```

**Cleanup sequence:**
1. Frees context with `llama_free(context_)`
2. Frees model with `llama_model_free(model_)`
3. Nullifies pointers

**Note:** Does not call `llama_backend_free()` (global cleanup, should only be called at application exit)

### Single Embedding Generation

```cpp
std::vector<float> generateEmbedding(const std::string& text);
```

**Process:**
1. Tokenizes input text with `llama_tokenize()`
2. Creates batch with `llama_batch_init()` (single sequence, seq_id=0)
3. Encodes batch with `llama_encode()`
4. Extracts embedding with `llama_get_embeddings_seq(context_, 0)`
5. Applies MEAN pooling automatically (configured in context params)
6. Returns embedding vector

**Returns:** Vector of floats (size = embedding dimension, typically 768)

**Throws:**
- `TokenizationError` if tokenization fails
- `EncodingError` if encoding fails
- `EncodingError` if embedding extraction fails

### Batch Embedding Generation

```cpp
std::vector<std::vector<float>> generateEmbeddingsBatch(
    const std::vector<std::string>& texts
);
```

**Process:**
1. Tokenizes all input texts
2. Creates single batch with `llama_batch_init()` containing all sequences
3. Assigns each text a unique sequence ID (0, 1, 2, ...)
4. Encodes all sequences in one pass with `llama_encode()`
5. Extracts embeddings for each sequence with `llama_get_embeddings_seq()`
6. Returns vector of embedding vectors

**Performance:** More efficient than calling `generateEmbedding()` multiple times because:
- Single model pass for all texts
- Reduced overhead from tokenization and batch setup
- Better CPU/GPU utilization

**Returns:** Vector of embedding vectors (one per input text)

**Throws:**
- `std::invalid_argument` if texts vector is empty
- `TokenizationError` if tokenization fails
- `EncodingError` if encoding fails
- `EncodingError` if embedding extraction fails

### Metadata Accessors

```cpp
int32_t getEmbeddingDim() const;   // Returns n_embd (e.g., 768)
int32_t getVocabSize() const;      // Returns n_vocab (e.g., 262144)
uint32_t getContextSize() const;   // Returns n_ctx (e.g., 2048)
```

## Move Semantics

The class supports move construction and move assignment, but disables copy:

```cpp
// Disabled (prevents double-free)
LlamaModel(const LlamaModel&) = delete;
LlamaModel& operator=(const LlamaModel&) = delete;

// Enabled (allows transfer of ownership)
LlamaModel(LlamaModel&& other) noexcept;
LlamaModel& operator=(LlamaModel&& other) noexcept;
```

**Example:**
```cpp
LlamaModel createModel(const std::string& path) {
    return LlamaModel(path);  // Move construction
}

LlamaModel model1("model1.gguf");
LlamaModel model2 = std::move(model1);  // Move assignment
// model1 is now in a moved-from state (safe but unusable)
```

## Thread Safety

**⚠️ IMPORTANT: Models are NOT thread-safe**

Each thread must create its own `LlamaModel` instance. Do not share a single instance across multiple threads.

**Correct usage:**
```cpp
void worker(const std::string& model_path, const std::string& text) {
    LlamaModel model(model_path);  // Each thread creates its own instance
    auto embedding = model.generateEmbedding(text);
    // ...
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker, "model.gguf", "text");
    }
    for (auto& t : threads) t.join();
}
```

**Incorrect usage:**
```cpp
LlamaModel model("model.gguf");  // ❌ Shared across threads - NOT SAFE

std::vector<std::thread> threads;
for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&model]() {
        model.generateEmbedding("text");  // ❌ Race condition
    });
}
```

## Exception Types

The library defines three custom exception types (all inherit from `std::runtime_error`):

- **`ModelError`**: Model loading or context initialization failures
- **`TokenizationError`**: Text tokenization failures
- **`EncodingError`**: Encoding failures (llama_encode errors, embedding extraction failures)

**Example exception handling:**
```cpp
try {
    LlamaModel model("path/to/model.gguf");
    auto embedding = model.generateEmbedding("Hello world");
} catch (const embedding::ModelError& e) {
    std::cerr << "Model error: " << e.what() << std::endl;
} catch (const embedding::TokenizationError& e) {
    std::cerr << "Tokenization error: " << e.what() << std::endl;
} catch (const embedding::EncodingError& e) {
    std::cerr << "Encoding error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Unknown error: " << e.what() << std::endl;
}
```

## Implementation Details

### Pooling Type

The wrapper uses **`LLAMA_POOLING_TYPE_MEAN`** (hardcoded), which:
- Averages token embeddings across the sequence
- Matches the Go implementation behavior
- Is the default for EmbeddingGemma models

### Tokenization

- Uses `llama_tokenize()` with `add_special=true` (adds BOS/EOS tokens)
- Automatically resizes buffer if initial size is insufficient
- Throws `TokenizationError` on failure

### Batch Structure

The `llama_batch` API requires:
- `n_tokens`: Total number of tokens across all sequences
- `token`: Array of token IDs
- `pos`: Array of token positions within each sequence
- `seq_id`: Array of sequence IDs (for batch processing)
- `n_seq_id`: Number of sequence IDs per token (always 1 in this implementation)
- `logits`: Array of logit flags (set to 0 since we only need embeddings)

### Context Configuration

- `n_ctx`: Maximum context size (0 = use model default, typically 2048)
- `n_batch`: Logical batch size (512 by default)
- `n_seq_max`: Maximum parallel sequences (64 - supports batch processing)
- `embeddings`: Enabled (true)
- `pooling_type`: MEAN pooling
- `n_threads`: Auto-detected if 0

## Performance Considerations

### Single vs Batch

| Operation | Overhead | Model Passes | Best For |
|-----------|----------|--------------|----------|
| Single | Low | 1 per text | Small batches (1-5 texts) |
| Batch | Moderate | 1 total | Large batches (5+ texts) |

**Recommendation:**
- Use `generateEmbedding()` for 1-5 texts
- Use `generateEmbeddingsBatch()` for 5+ texts

### Memory Usage

- Model size: ~307 MB (Q8_0 quantized EmbeddingGemma)
- Context buffer: ~80 MB (n_seq_max=64)
- Per-embedding output: 768 floats × 4 bytes = 3 KB

### Backend Initialization

The static member `backend_initialized_` ensures `llama_backend_init()` is called exactly once, even if multiple `LlamaModel` instances are created.

## Example Usage

See `src/main.cpp` for comprehensive examples including:
- Single embedding generation
- Batch embedding generation
- Cosine similarity computation
- Reproducibility testing
- Error handling

## Building

```bash
cd benchmark_cpp
LLAMA_CPP_PATH=/path/to/llama.cpp make
```

## Testing

```bash
./build/benchmark_cpp model/embeddinggemma-300m-Q8_0.gguf
```

Expected output:
- Model metadata (dimension, vocab size, context size)
- Single embedding test (768 dimensions, L2 norm ~44)
- Batch embedding test (3 texts, varying L2 norms)
- Similarity scores (cosine similarity between text pairs)
- Reproducibility test (identical texts produce identical embeddings)

## Compatibility

- **C++ Standard**: C++17 or later
- **llama.cpp**: Latest main branch (tested with commit as of 2025-01-27)
- **Platform**: Linux, macOS, Windows (cross-platform)
- **Compiler**: GCC 7+, Clang 5+, MSVC 2019+

## Known Limitations

1. **Thread safety**: Models must not be shared across threads
2. **Pooling type**: Hardcoded to MEAN (cannot change at runtime)
3. **Batch size**: Limited to 64 parallel sequences (configurable via n_seq_max)
4. **Context size**: Limited by model's maximum (typically 2048 tokens)

## Future Enhancements

Possible improvements:
- [ ] Configurable pooling type (MEAN, CLS, LAST)
- [ ] Async API for non-blocking embedding generation
- [ ] Streaming API for large texts
- [ ] GPU offloading support (n_gpu_layers parameter)
- [ ] Model caching across multiple instances
- [ ] Thread pool for batch processing
