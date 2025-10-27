#include "benchmark_isolated.h"
#include "model.h"
#include "test_data.h"
#include "metrics.h"
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace benchmark {

/**
 * Worker function that runs in a separate thread.
 * Each worker loads its own model instance and generates embeddings until stopped.
 */
static void workerFunc(
    const std::string& model_path,
    int worker_id,
    std::atomic<bool>& stop_flag,
    WorkerResult& result)
{
    try {
        // Load model instance (independent per worker)
        embedding::LlamaModel model(model_path);

        std::cerr << "Worker " << worker_id << ": model loaded, starting benchmark\n";

        int count = 0;

        // Generate embeddings until stop flag is set
        while (!stop_flag.load(std::memory_order_acquire)) {
            // Get random text from corpus
            const std::string& text = getRandomText();

            // Generate single embedding
            try {
                auto embedding = model.generateEmbedding(text);
                count++;
            } catch (const std::exception& e) {
                // Log error but continue
                std::cerr << "Worker " << worker_id << ": embedding failed: "
                          << e.what() << "\n";
            }
        }

        // Store result
        result.worker_id = worker_id;
        result.embeddings_count = count;

        std::cerr << "Worker " << worker_id << ": completed " << count << " embeddings\n";

    } catch (const std::exception& e) {
        std::cerr << "Worker " << worker_id << ": failed to load model: "
                  << e.what() << "\n";
        result.worker_id = worker_id;
        result.embeddings_count = 0;
    }
}

std::vector<WorkerResult> runIsolatedMode(
    const std::string& model_path,
    int duration_sec,
    int num_workers)
{
    // Create stop flag (atomic for thread safety)
    std::atomic<bool> stop_flag(false);

    // Create results vector
    std::vector<WorkerResult> results(num_workers);

    // Create thread vector
    std::vector<std::thread> threads;
    threads.reserve(num_workers);

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(
            workerFunc,
            std::cref(model_path),
            i,
            std::ref(stop_flag),
            std::ref(results[i])
        );
    }

    // Sleep for duration
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));

    // Set stop flag to signal workers to stop
    stop_flag.store(true, std::memory_order_release);

    // Join all threads (wait for them to complete)
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Calculate actual duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double duration_seconds = duration.count() / 1e9;

    // Calculate aggregate metrics
    int total_embeddings = 0;
    int min_count = results[0].embeddings_count;
    int max_count = results[0].embeddings_count;

    for (const auto& result : results) {
        total_embeddings += result.embeddings_count;
        if (result.embeddings_count < min_count) {
            min_count = result.embeddings_count;
        }
        if (result.embeddings_count > max_count) {
            max_count = result.embeddings_count;
        }
    }

    double avg_count = static_cast<double>(total_embeddings) / num_workers;
    double throughput = total_embeddings / duration_seconds;
    double avg_latency_ms = (duration_seconds * 1000.0) / total_embeddings;

    // Print benchmark results
    metrics::OutputFormatter::printBenchmarkResults(
        "isolated",
        duration_seconds,
        total_embeddings,
        throughput,
        avg_latency_ms
    );

    // Print per-worker statistics
    metrics::OutputFormatter::printWorkerStats(
        num_workers,
        min_count,
        max_count,
        avg_count
    );

    // Get memory statistics (RSS)
    metrics::MemoryStats mem_stats;
    double rss_mb = mem_stats.getRssMB();

    // Print memory statistics (using RSS as proxy for HeapAlloc, no GC in C++)
    metrics::OutputFormatter::printMemoryStats(
        rss_mb,    // HeapAlloc (approximated by RSS)
        rss_mb,    // TotalAlloc (same as HeapAlloc for C++)
        0          // NumGC (no GC in C++)
    );

    return results;
}

} // namespace benchmark
