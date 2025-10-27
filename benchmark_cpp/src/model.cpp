#include "model.h"
#include <cstring>
#include <sstream>
#include <algorithm>

namespace embedding {

// Static member initialization
bool LlamaModel::backend_initialized_ = false;

void LlamaModel::initBackend() {
    if (!backend_initialized_) {
        llama_backend_init();
        backend_initialized_ = true;
    }
}

LlamaModel::LlamaModel(
    const std::string& model_path,
    uint32_t n_ctx,
    uint32_t n_batch,
    int32_t n_threads
)
    : model_(nullptr)
    , context_(nullptr)
    , vocab_(nullptr)
    , n_embd_(0)
    , n_vocab_(0)
    , n_ctx_(0)
{
    // Initialize backend (idempotent - only runs once)
    initBackend();

    // Initialize model parameters with defaults
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.vocab_only = false;

    // Load model from file
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        throw ModelError("Failed to load model from: " + model_path);
    }

    // Get vocabulary handle
    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw ModelError("Failed to get vocabulary from model");
    }

    // Cache model parameters
    n_embd_ = llama_model_n_embd(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    // Initialize context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;  // 0 = use model default
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch;  // Physical batch size (must be >= n_batch for encoder models)
    ctx_params.n_threads = n_threads;  // 0 = auto-detect
    ctx_params.n_threads_batch = n_threads;  // 0 = auto-detect
    ctx_params.n_seq_max = 256;  // Support up to 256 parallel sequences for batch processing

    // Enable embeddings with MEAN pooling (matching Go implementation)
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    // Create context
    context_ = llama_init_from_model(model_, ctx_params);
    if (!context_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw ModelError("Failed to initialize context from model");
    }

    // Cache context size
    n_ctx_ = llama_n_ctx(context_);
}

LlamaModel::~LlamaModel() {
    cleanup();
}

void LlamaModel::cleanup() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
    // Note: We don't call llama_backend_free() because it's a global cleanup
    // that should only be called when the entire application exits
}

LlamaModel::LlamaModel(LlamaModel&& other) noexcept
    : model_(other.model_)
    , context_(other.context_)
    , vocab_(other.vocab_)
    , n_embd_(other.n_embd_)
    , n_vocab_(other.n_vocab_)
    , n_ctx_(other.n_ctx_)
{
    other.model_ = nullptr;
    other.context_ = nullptr;
    other.vocab_ = nullptr;
}

LlamaModel& LlamaModel::operator=(LlamaModel&& other) noexcept {
    if (this != &other) {
        cleanup();

        model_ = other.model_;
        context_ = other.context_;
        vocab_ = other.vocab_;
        n_embd_ = other.n_embd_;
        n_vocab_ = other.n_vocab_;
        n_ctx_ = other.n_ctx_;

        other.model_ = nullptr;
        other.context_ = nullptr;
        other.vocab_ = nullptr;
    }
    return *this;
}

std::vector<llama_token> LlamaModel::tokenize(
    const std::string& text,
    bool add_special
) {
    // First pass: determine required buffer size
    const int32_t text_len = static_cast<int32_t>(text.size());
    const int32_t n_tokens_max = text_len + (add_special ? 8 : 4);  // Conservative estimate

    std::vector<llama_token> tokens(n_tokens_max);

    const int32_t n_tokens = llama_tokenize(
        vocab_,
        text.c_str(),
        text_len,
        tokens.data(),
        n_tokens_max,
        add_special,
        false  // parse_special = false
    );

    if (n_tokens < 0) {
        // Need larger buffer
        const int32_t required_size = -n_tokens;
        tokens.resize(required_size);

        const int32_t n_tokens_retry = llama_tokenize(
            vocab_,
            text.c_str(),
            text_len,
            tokens.data(),
            required_size,
            add_special,
            false
        );

        if (n_tokens_retry < 0) {
            throw TokenizationError("Failed to tokenize text after buffer resize");
        }
        tokens.resize(n_tokens_retry);
    } else {
        tokens.resize(n_tokens);
    }

    if (tokens.empty()) {
        throw TokenizationError("Tokenization produced empty token sequence");
    }

    return tokens;
}

std::vector<float> LlamaModel::generateEmbedding(const std::string& text) {
    // Tokenize input text
    std::vector<llama_token> tokens = tokenize(text, true);

    // Create batch with single sequence
    llama_batch batch = llama_batch_init(
        static_cast<int32_t>(tokens.size()),
        0,  // embd = 0 (using token IDs, not embeddings)
        1   // n_seq_max = 1 (single sequence)
    );

    // Fill batch with tokens
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // sequence ID = 0
        batch.logits[i] = 0;  // we only need embeddings, not logits
    }

    // Encode batch
    const int32_t result = llama_encode(context_, batch);

    // Free batch
    llama_batch_free(batch);

    if (result != 0) {
        std::ostringstream oss;
        oss << "llama_encode failed with code: " << result;
        throw EncodingError(oss.str());
    }

    // Extract embedding for sequence 0 (MEAN pooling applied automatically)
    const float* embedding_ptr = llama_get_embeddings_seq(context_, 0);
    if (!embedding_ptr) {
        throw EncodingError("Failed to get embeddings for sequence");
    }

    // Copy embedding to output vector
    std::vector<float> embedding(embedding_ptr, embedding_ptr + n_embd_);

    return embedding;
}

std::vector<std::vector<float>> LlamaModel::generateEmbeddingsBatch(
    const std::vector<std::string>& texts
) {
    if (texts.empty()) {
        throw std::invalid_argument("texts vector cannot be empty");
    }

    // Tokenize all texts
    std::vector<std::vector<llama_token>> all_tokens;
    all_tokens.reserve(texts.size());
    int32_t total_tokens = 0;

    for (const auto& text : texts) {
        auto tokens = tokenize(text, true);
        total_tokens += static_cast<int32_t>(tokens.size());
        all_tokens.push_back(std::move(tokens));
    }

    // Create batch with multiple sequences
    llama_batch batch = llama_batch_init(
        total_tokens,
        0,  // embd = 0 (using token IDs, not embeddings)
        static_cast<int32_t>(texts.size())  // n_seq_max
    );

    // Fill batch with tokens from all sequences
    int32_t token_idx = 0;
    for (size_t seq_idx = 0; seq_idx < all_tokens.size(); ++seq_idx) {
        const auto& tokens = all_tokens[seq_idx];

        for (size_t pos = 0; pos < tokens.size(); ++pos) {
            batch.token[token_idx] = tokens[pos];
            batch.pos[token_idx] = static_cast<llama_pos>(pos);
            batch.n_seq_id[token_idx] = 1;
            batch.seq_id[token_idx][0] = static_cast<llama_seq_id>(seq_idx);
            batch.logits[token_idx] = 0;  // we only need embeddings, not logits
            ++token_idx;
        }
    }
    batch.n_tokens = total_tokens;

    // Encode batch (all sequences in one pass)
    const int32_t result = llama_encode(context_, batch);

    // Free batch
    llama_batch_free(batch);

    if (result != 0) {
        std::ostringstream oss;
        oss << "llama_encode failed with code: " << result;
        throw EncodingError(oss.str());
    }

    // Extract embeddings for all sequences
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(texts.size());

    for (size_t seq_idx = 0; seq_idx < texts.size(); ++seq_idx) {
        const float* embedding_ptr = llama_get_embeddings_seq(
            context_,
            static_cast<llama_seq_id>(seq_idx)
        );

        if (!embedding_ptr) {
            std::ostringstream oss;
            oss << "Failed to get embeddings for sequence " << seq_idx;
            throw EncodingError(oss.str());
        }

        embeddings.emplace_back(embedding_ptr, embedding_ptr + n_embd_);
    }

    return embeddings;
}

int32_t LlamaModel::getEmbeddingDim() const {
    return n_embd_;
}

int32_t LlamaModel::getVocabSize() const {
    return n_vocab_;
}

uint32_t LlamaModel::getContextSize() const {
    return n_ctx_;
}

} // namespace embedding
