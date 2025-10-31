#include "test_data.h"
#include <iostream>
#include <cassert>
#include <cstring>

// This test validates that C++ test data exactly matches the Go benchmark data
// Run this to verify character-for-character matching

namespace {

// Expected values from Go benchmark (cmd/benchmark/main.go)
const char* EXPECTED_SHORT_DOC = "The quick brown fox jumps over the lazy dog";

const char* EXPECTED_LONG_DOC = "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.";

const char* EXPECTED_SHORT_DOCS[] = {
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming modern technology",
    "Machine learning enables computers to learn from data",
    "Neural networks process information efficiently",
    "Deep learning powers many AI applications"
};

const char* EXPECTED_TEST_TEXTS[] = {
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

void testShortDoc() {
    std::cout << "Testing SHORT_DOC... ";
    assert(std::strcmp(benchmark::SHORT_DOC, EXPECTED_SHORT_DOC) == 0);
    assert(std::strcmp(benchmark::getShortDoc(), EXPECTED_SHORT_DOC) == 0);
    std::cout << "PASS\n";
}

void testLongDoc() {
    std::cout << "Testing LONG_DOC... ";
    assert(std::strcmp(benchmark::LONG_DOC, EXPECTED_LONG_DOC) == 0);
    assert(std::strcmp(benchmark::getLongDoc(), EXPECTED_LONG_DOC) == 0);
    std::cout << "PASS\n";
}

void testShortDocs() {
    std::cout << "Testing SHORT_DOCS... ";
    assert(benchmark::SHORT_DOCS.size() == 5);
    assert(benchmark::getShortDocsSize() == 5);

    for (std::size_t i = 0; i < 5; ++i) {
        assert(benchmark::SHORT_DOCS[i] == EXPECTED_SHORT_DOCS[i]);
    }
    std::cout << "PASS\n";
}

void testTestTexts() {
    std::cout << "Testing TEST_TEXTS... ";
    assert(benchmark::TEST_TEXTS.size() == 10);
    assert(benchmark::getTestTextsSize() == 10);

    for (std::size_t i = 0; i < 10; ++i) {
        assert(benchmark::TEST_TEXTS[i] == EXPECTED_TEST_TEXTS[i]);
    }
    std::cout << "PASS\n";
}

void testHelperFunctions() {
    std::cout << "Testing helper functions... ";

    // Test getRandomText returns valid references
    for (int i = 0; i < 100; ++i) {
        const std::string& text = benchmark::getRandomText();
        bool found = false;
        for (const auto& expected : benchmark::TEST_TEXTS) {
            if (text == expected) {
                found = true;
                break;
            }
        }
        assert(found);
    }

    // Test getRandomShortDoc returns valid references
    for (int i = 0; i < 100; ++i) {
        const std::string& text = benchmark::getRandomShortDoc();
        bool found = false;
        for (const auto& expected : benchmark::SHORT_DOCS) {
            if (text == expected) {
                found = true;
                break;
            }
        }
        assert(found);
    }

    std::cout << "PASS\n";
}

void printTextStats() {
    std::cout << "\n=== Text Statistics ===\n";

    // Count words in SHORT_DOC
    int shortWords = 1;
    for (const char* p = benchmark::SHORT_DOC; *p; ++p) {
        if (*p == ' ') shortWords++;
    }
    std::cout << "SHORT_DOC: " << shortWords << " words, "
              << std::strlen(benchmark::SHORT_DOC) << " characters\n";

    // Count words in LONG_DOC
    int longWords = 1;
    for (const char* p = benchmark::LONG_DOC; *p; ++p) {
        if (*p == ' ') longWords++;
    }
    std::cout << "LONG_DOC: " << longWords << " words, "
              << std::strlen(benchmark::LONG_DOC) << " characters\n";

    // Count words in SHORT_DOCS
    std::cout << "\nSHORT_DOCS collection (" << benchmark::SHORT_DOCS.size() << " documents):\n";
    for (std::size_t i = 0; i < benchmark::SHORT_DOCS.size(); ++i) {
        int words = 1;
        for (char c : benchmark::SHORT_DOCS[i]) {
            if (c == ' ') words++;
        }
        std::cout << "  [" << i << "] " << words << " words, "
                  << benchmark::SHORT_DOCS[i].length() << " characters\n";
    }

    // Count words in TEST_TEXTS
    std::cout << "\nTEST_TEXTS corpus (" << benchmark::TEST_TEXTS.size() << " documents):\n";
    for (std::size_t i = 0; i < benchmark::TEST_TEXTS.size(); ++i) {
        int words = 1;
        for (char c : benchmark::TEST_TEXTS[i]) {
            if (c == ' ') words++;
        }
        std::cout << "  [" << i << "] " << words << " words, "
                  << benchmark::TEST_TEXTS[i].length() << " characters\n";
    }
}

} // anonymous namespace

int main() {
    std::cout << "=== C++ Test Data Validation ===\n";
    std::cout << "Validating exact match with Go benchmark data\n";
    std::cout << "(cmd/benchmark/main.go)\n\n";

    try {
        testShortDoc();
        testLongDoc();
        testShortDocs();
        testTestTexts();
        testHelperFunctions();

        std::cout << "\n=== ALL TESTS PASSED ===\n";
        std::cout << "C++ test data matches Go benchmark exactly (character-for-character)\n";

        printTextStats();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nFAILED: " << e.what() << "\n";
        return 1;
    }
}
