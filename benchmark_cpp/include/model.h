#ifndef MODEL_H
#define MODEL_H

#include "llama.h"
#include <string>
#include <vector>
#include <stdexcept>

namespace embedding {

/**
 * RAII wrapper for llama.cpp model and context management with embedding support.
 *
 * This class provides a safe, exception-based interface for loading GGUF models
 * and generating embeddings using the llama.cpp library.
 *
 * Thread Safety:
 *   Models are NOT thread-safe. Each thread must create its own LlamaModel instance.
 *   Do not share a single LlamaModel instance across multiple threads.
 *
 * Usage:
 *   try {
 *       LlamaModel model("path/to/model.gguf");
 *       auto embedding = model.generateEmbedding("Hello world");
 *
 *       std::vector<std::string> texts = {"text1", "text2", "text3"};
 *       auto embeddings = model.generateEmbeddingsBatch(texts);
 *   } catch (const std::exception& e) {
 *       std::cerr << "Error: " << e.what() << std::endl;
 *   }
 */
class LlamaModel {
public:
    /**
     * Load a GGUF model from the specified path.
     *
     * @param model_path Path to the GGUF model file
     * @param n_ctx Context size (0 = use model default)
     * @param n_batch Batch size for processing
     * @param n_threads Number of threads for generation (0 = auto-detect)
     *
     * @throws std::runtime_error if model loading or context initialization fails
     */
    explicit LlamaModel(
        const std::string& model_path,
        uint32_t n_ctx = 0,
        uint32_t n_batch = 16384,
        int32_t n_threads = 0
    );

    /**
     * Destructor: properly frees context and model resources.
     * Guaranteed to clean up even if exceptions occurred during construction.
     */
    ~LlamaModel();

    // Disable copy constructor and copy assignment (prevent double-free)
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    // Enable move constructor and move assignment (allow transfer of ownership)
    LlamaModel(LlamaModel&& other) noexcept;
    LlamaModel& operator=(LlamaModel&& other) noexcept;

    /**
     * Generate embedding for a single text.
     *
     * Uses LLAMA_POOLING_TYPE_MEAN pooling (matching Go implementation).
     *
     * @param text Input text to embed
     * @return Embedding vector (size = model's embedding dimension)
     *
     * @throws std::runtime_error if tokenization fails
     * @throws std::runtime_error if encoding fails
     * @throws std::runtime_error if embedding extraction fails
     */
    std::vector<float> generateEmbedding(const std::string& text);

    /**
     * Generate embeddings for multiple texts in a single batch.
     *
     * This method is more efficient than calling generateEmbedding() multiple times
     * because it uses the llama_batch API to encode all sequences in one pass.
     *
     * Uses LLAMA_POOLING_TYPE_MEAN pooling (matching Go implementation).
     *
     * @param texts Vector of input texts to embed
     * @return Vector of embedding vectors (one per input text)
     *
     * @throws std::runtime_error if tokenization fails
     * @throws std::runtime_error if encoding fails
     * @throws std::runtime_error if embedding extraction fails
     * @throws std::invalid_argument if texts vector is empty
     */
    std::vector<std::vector<float>> generateEmbeddingsBatch(
        const std::vector<std::string>& texts
    );

    /**
     * Get the embedding dimension of the loaded model.
     *
     * @return Number of dimensions in each embedding vector
     */
    int32_t getEmbeddingDim() const;

    /**
     * Get the vocabulary size of the loaded model.
     *
     * @return Number of tokens in the vocabulary
     */
    int32_t getVocabSize() const;

    /**
     * Get the maximum context size supported by the model.
     *
     * @return Maximum number of tokens the model can process
     */
    uint32_t getContextSize() const;

private:
    /**
     * Tokenize a text string.
     *
     * @param text Input text to tokenize
     * @param add_special Whether to add special tokens (BOS/EOS)
     * @return Vector of token IDs
     *
     * @throws std::runtime_error if tokenization fails
     */
    std::vector<llama_token> tokenize(
        const std::string& text,
        bool add_special = true
    );

    /**
     * Initialize the llama.cpp backend (called once in constructor).
     */
    static void initBackend();

    /**
     * Clean up resources. Called by destructor and move assignment.
     */
    void cleanup();

    // Model and context handles
    llama_model* model_;
    llama_context* context_;
    const llama_vocab* vocab_;

    // Model parameters (cached for convenience)
    int32_t n_embd_;
    int32_t n_vocab_;
    uint32_t n_ctx_;

    // Backend initialization tracking
    static bool backend_initialized_;
};

/**
 * Exception class for model-specific errors.
 */
class ModelError : public std::runtime_error {
public:
    explicit ModelError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * Exception class for tokenization errors.
 */
class TokenizationError : public std::runtime_error {
public:
    explicit TokenizationError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * Exception class for encoding errors.
 */
class EncodingError : public std::runtime_error {
public:
    explicit EncodingError(const std::string& message)
        : std::runtime_error(message) {}
};

} // namespace embedding

#endif // MODEL_H
