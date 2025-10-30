#ifndef BENCHMARK_BATCH_H
#define BENCHMARK_BATCH_H

#include <string>

namespace benchmark {

/**
 * Run batch mode benchmark matching Go implementation.
 *
 * This benchmark:
 * - Uses a single shared model instance across all iterations
 * - Runs for the specified duration (seconds)
 * - Each iteration randomly selects batchSize texts from TEST_TEXTS corpus
 * - Processes batches using generateEmbeddingsBatch()
 * - Tracks throughput (embeddings/sec) and latency metrics
 * - Reports memory statistics at the end
 *
 * Output format matches Go's "=== Benchmark Results ===" format.
 *
 * @param modelPath Path to the GGUF model file
 * @param duration Duration to run the benchmark in seconds
 * @param batchSize Number of texts to process in each batch
 * @param threads llama.cpp CPU threads per context
 * @return 0 on success, non-zero on failure
 */
int runBatchMode(const std::string& modelPath, int duration, int batchSize, int threads);

} // namespace benchmark

#endif // BENCHMARK_BATCH_H
