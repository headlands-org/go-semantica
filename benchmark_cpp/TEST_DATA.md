# C++ Test Data Module

This module provides identical test texts and corpus as the Go benchmark for ensuring consistent benchmark comparisons between the Go implementation and llama.cpp C++ implementation.

## Files

- `include/test_data.h` - Header file with constant declarations and helper functions
- `src/test_data.cpp` - Implementation with exact text data from Go benchmark
- `src/test_data_test.cpp` - Validation test ensuring character-for-character match

## Test Data

All test data is extracted from `/cmd/benchmark/main.go` to ensure exact matching:

### 1. SHORT_DOC (9 words)
Single short document for latency testing:
```
"The quick brown fox jumps over the lazy dog"
```

### 2. LONG_DOC (49 words)
Single long document for latency testing:
```
"Machine learning has revolutionized artificial intelligence by enabling computers
to learn from data without explicit programming. Modern neural networks can process
vast amounts of information and identify complex patterns that would be impossible
for humans to detect manually. This technology powers applications from image
recognition to natural language processing."
```

### 3. SHORT_DOCS (5 documents)
Collection of 5 varied short sentences (6-9 words each) for batch testing:
- "The quick brown fox jumps over the lazy dog"
- "Artificial intelligence is transforming modern technology"
- "Machine learning enables computers to learn from data"
- "Neural networks process information efficiently"
- "Deep learning powers many AI applications"

### 4. TEST_TEXTS (10 documents)
Technical corpus for batch/isolated mode benchmarking:
1. "Machine learning enables computers to learn from data without explicit programming"
2. "Neural networks use layers of interconnected nodes to process complex patterns"
3. "Distributed systems coordinate multiple computers to solve problems at scale"
4. "Encryption algorithms protect sensitive data through mathematical transformations"
5. "Quantum computing leverages quantum mechanics to solve certain problems exponentially faster"
6. "Graph databases optimize storage and retrieval of highly connected data"
7. "Microservices architecture decomposes applications into independent deployable services"
8. "Container orchestration automates deployment scaling and management of containerized applications"
9. "Natural language processing enables computers to understand and generate human language"
10. "Computer vision algorithms extract meaningful information from digital images and videos"

## API Functions

```cpp
namespace benchmark {
    // Get single documents
    const char* getShortDoc();        // Returns SHORT_DOC
    const char* getLongDoc();         // Returns LONG_DOC

    // Get random samples
    const std::string& getRandomText();      // Random from TEST_TEXTS
    const std::string& getRandomShortDoc();  // Random from SHORT_DOCS

    // Get corpus sizes
    std::size_t getTestTextsSize();    // Returns 10
    std::size_t getShortDocsSize();    // Returns 5

    // Direct access to collections
    extern const std::vector<std::string> SHORT_DOCS;
    extern const std::vector<std::string> TEST_TEXTS;
}
```

## Usage Example

```cpp
#include "test_data.h"
#include <iostream>

int main() {
    // Get single documents for latency tests
    auto shortDoc = benchmark::getShortDoc();
    auto longDoc = benchmark::getLongDoc();

    // Get random texts for throughput tests
    for (int i = 0; i < 100; ++i) {
        const auto& text = benchmark::getRandomText();
        // Process text...
    }

    // Access full corpus
    for (const auto& text : benchmark::TEST_TEXTS) {
        // Process text...
    }

    return 0;
}
```

## Building and Testing

### Build the validation test:
```bash
cd benchmark_cpp
make test
```

### Run the validation test:
```bash
./build/test_data_test
```

The test validates:
- ✓ SHORT_DOC matches exactly (9 words, 43 characters)
- ✓ LONG_DOC matches exactly (49 words, 370 characters)
- ✓ SHORT_DOCS collection matches exactly (5 documents, 6-9 words each)
- ✓ TEST_TEXTS corpus matches exactly (10 documents, 8-11 words each)
- ✓ Helper functions return valid references
- ✓ Character-for-character matching with Go implementation

## Validation Results

```
=== ALL TESTS PASSED ===
C++ test data matches Go benchmark exactly (character-for-character)

SHORT_DOC: 9 words, 43 characters
LONG_DOC: 49 words, 370 characters
SHORT_DOCS: 5 documents (6-9 words each)
TEST_TEXTS: 10 documents (8-11 words each)
```

## Design Notes

- **Zero-copy design**: Direct access via const references where possible
- **Thread-safe random selection**: Each helper function uses its own random generator
- **No external dependencies**: Pure C++17 standard library
- **Namespace isolation**: All symbols in `benchmark` namespace to avoid conflicts
- **Exact matching**: Every character matches the Go implementation byte-for-byte

## Benchmark Usage

This module is designed for the 5-scenario benchmark matrix:

1. **Idle Memory** - Model load only
2. **Single Short Doc (9w)** - Use `getShortDoc()`
3. **Single Long Doc (49w)** - Use `getLongDoc()`
4. **Batch Short Docs (96x)** - Use `getRandomShortDoc()` or iterate `SHORT_DOCS`
5. **Batch Long Docs (96x)** - Use `getLongDoc()` repeated

## Maintenance

When updating test data:
1. Update `cmd/benchmark/main.go` in Go codebase (source of truth)
2. Update `benchmark_cpp/src/test_data.cpp` to match exactly
3. Run `make test` to verify character-for-character matching
4. Ensure word counts match (comments indicate expected counts)
