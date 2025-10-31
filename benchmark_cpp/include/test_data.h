#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <string>
#include <vector>
#include <cstddef>

namespace benchmark {

// Short document (9 words) - used for single short doc latency tests
extern const char* SHORT_DOC;

// Long document (49 words) - used for single long doc latency tests
extern const char* LONG_DOC;

// Extra-long document (~400 words) - long doc repeated 8 times
extern const char* EXTRA_LONG_DOC;

// Short documents for batch tests (5 varied sentences, 6-9 words each)
extern const std::vector<std::string> SHORT_DOCS;

// Test corpus (10 technical documents) - used for batch/isolated modes
extern const std::vector<std::string> TEST_TEXTS;

// Helper functions

// Get the short document for single doc tests
const char* getShortDoc();

// Get the long document for single doc tests
const char* getLongDoc();

// Get the extra-long document for single doc tests
const char* getExtraLongDoc();

// Get a random text from the test corpus
const std::string& getRandomText();

// Get a random short doc from the short docs collection
const std::string& getRandomShortDoc();

// Get the size of the test corpus
std::size_t getTestTextsSize();

// Get the size of the short docs collection
std::size_t getShortDocsSize();

} // namespace benchmark

#endif // TEST_DATA_H
