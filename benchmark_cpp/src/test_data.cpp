#include "test_data.h"
#include <random>

namespace benchmark {

// Short document (9 words) - exact match from Go benchmark
const char* SHORT_DOC = "The quick brown fox jumps over the lazy dog";

// Long document (49 words) - exact match from Go benchmark
const char* LONG_DOC = "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.";

// Extra-long document (~400 words) - long doc repeated 8 times
const char* EXTRA_LONG_DOC = "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing. Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.";

// Short documents for batch tests (5 varied sentences, 6-9 words each)
// Exact match from Go shortDocs array
const std::vector<std::string> SHORT_DOCS = {
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming modern technology",
    "Machine learning enables computers to learn from data",
    "Neural networks process information efficiently",
    "Deep learning powers many AI applications"
};

// Test corpus (10 technical documents) - exact match from Go testTexts array
const std::vector<std::string> TEST_TEXTS = {
    "Machine learning enables computers to learn from data without explicit programming",
    "Neural networks use layers of interconnected nodes to process complex patterns",
    "Distributed systems coordinate multiple computers to solve problems at scale",
    "Encryption algorithms protect sensitive data through mathematical transformations",
    "Quantum computing leverages quantum mechanics to solve certain problems exponentially faster",
    "Graph databases optimize storage and retrieval of highly connected data",
    "Microservices architecture decomposes applications into independent deployable services",
    "Container orchestration automates deployment scaling and management of containerized applications",
    "Natural language processing enables computers to understand and generate human language",
    "Computer vision algorithms extract meaningful information from digital images and videos"
};

// Helper functions

const char* getShortDoc() {
    return SHORT_DOC;
}

const char* getLongDoc() {
    return LONG_DOC;
}

const char* getExtraLongDoc() {
    return EXTRA_LONG_DOC;
}

const std::string& getRandomText() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<std::size_t> dis(0, TEST_TEXTS.size() - 1);
    return TEST_TEXTS[dis(gen)];
}

const std::string& getRandomShortDoc() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<std::size_t> dis(0, SHORT_DOCS.size() - 1);
    return SHORT_DOCS[dis(gen)];
}

std::size_t getTestTextsSize() {
    return TEST_TEXTS.size();
}

std::size_t getShortDocsSize() {
    return SHORT_DOCS.size();
}

} // namespace benchmark
