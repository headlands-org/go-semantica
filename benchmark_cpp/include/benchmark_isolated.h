#ifndef BENCHMARK_ISOLATED_H
#define BENCHMARK_ISOLATED_H

#include <string>
#include <vector>

namespace benchmark {

/**
 * Per-worker result structure
 */
struct WorkerResult {
    int worker_id;
    int embeddings_count;
};

/**
 * Run isolated mode benchmark with independent worker threads.
 *
 * Each worker thread:
 * - Loads its own model instance (no sharing)
 * - Runs in parallel using std::thread
 * - Generates single embeddings (not batched) until duration expires
 *
 * @param model_path Path to GGUF model file
 * @param duration_sec Benchmark duration in seconds
 * @param num_workers Number of worker threads to spawn
 * @return Vector of per-worker results
 */
std::vector<WorkerResult> runIsolatedMode(
    const std::string& model_path,
    int duration_sec,
    int num_workers
);

} // namespace benchmark

#endif // BENCHMARK_ISOLATED_H
