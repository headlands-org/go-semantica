#include "benchmark_comprehensive.h"
#include "model.h"
#include "metrics.h"
#include "test_data.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>

namespace benchmark {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Measure RSS memory in MB.
 */
static double measureMemoryMB() {
    metrics::MemoryStats mem;
    return mem.getRssMB();
}

/**
 * Run warmup embeddings to warm up caches.
 */
static void warmup(embedding::LlamaModel& model, const std::string& doc) {
    for (int i = 0; i < 5; i++) {
        try {
            model.generateEmbedding(doc);
        } catch (const std::exception& e) {
            std::cerr << "Warning: warmup failed: " << e.what() << std::endl;
        }
    }
}

/**
 * Measure single document latency with percentiles.
 * Performs 20 runs and calculates P50/P95/P99 latencies.
 */
static metrics::OutputFormatter::LatencyStats measureSingleDocLatency(
    embedding::LlamaModel& model,
    const std::string& doc)
{
    const int NUM_RUNS = 20;
    metrics::LatencyTracker tracker;

    for (int i = 0; i < NUM_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        try {
            model.generateEmbedding(doc);
        } catch (const std::exception& e) {
            std::cerr << "Warning: embedding failed during latency test: " << e.what() << std::endl;
            tracker.addMeasurementMs(0.0);
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        tracker.addMeasurement(end - start);
    }

    auto stats = tracker.calculate();
    return {stats.mean, stats.p50, stats.p95, stats.p99};
}

/**
 * Measure throughput with memory sampling.
 * Runs for specified duration, samples memory every 100ms, and tracks peak.
 */
static metrics::OutputFormatter::ThroughputStats measureThroughput(
    embedding::LlamaModel& model,
    const std::vector<std::string>& docs,
    int batch_size,
    int duration_sec)
{
    std::atomic<bool> done{false};
    std::atomic<double> peak_memory_mb{0.0};

    // Background thread to sample memory every 100ms
    std::thread memory_sampler([&done, &peak_memory_mb]() {
        while (!done.load()) {
            double current_mb = measureMemoryMB();
            double current_peak = peak_memory_mb.load();
            while (current_mb > current_peak &&
                   !peak_memory_mb.compare_exchange_weak(current_peak, current_mb)) {
                // Loop until we successfully update or another thread sets a higher value
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Run throughput test
    auto start_time = std::chrono::high_resolution_clock::now();
    auto deadline = start_time + std::chrono::seconds(duration_sec);
    int total_embeddings = 0;

    // Random number generator for selecting docs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, docs.size() - 1);

    while (std::chrono::high_resolution_clock::now() < deadline) {
        // Select batch_size random texts from docs
        std::vector<std::string> batch;
        batch.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
            batch.push_back(docs[dis(gen)]);
        }

        // Generate embeddings
        try {
            model.generateEmbeddingsBatch(batch);
            total_embeddings += batch.size();
        } catch (const std::exception& e) {
            std::cerr << "Warning: embedding failed during throughput test: " << e.what() << std::endl;
            continue;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double actual_duration = std::chrono::duration<double>(end_time - start_time).count();

    // Stop memory sampling
    done.store(true);
    memory_sampler.join();

    double throughput = static_cast<double>(total_embeddings) / actual_duration;
    double peak_mb = peak_memory_mb.load();

    return {throughput, peak_mb, actual_duration, total_embeddings};
}

// ============================================================================
// Main Comprehensive Benchmark Function
// ============================================================================

void runComprehensiveMode(const std::string& model_path) {
    std::cerr << "Running 5-scenario benchmark suite..." << std::endl;

    // 1. Platform detection
    std::cerr << "[1/6] Detecting platform..." << std::endl;
    metrics::OutputFormatter::PlatformInfo platform;
    platform.cpu = metrics::detectCPU();
    platform.cores = metrics::detectCores();
    platform.os = metrics::detectOS();
    platform.arch = metrics::detectArch();
    std::cerr << "Detected: " << platform.cpu << ", " << platform.cores
              << " cores, " << platform.os << "/" << platform.arch << std::endl;

    // 2. Load model and measure idle memory
    std::cerr << "[2/6] Loading model and measuring idle memory..." << std::endl;
    embedding::LlamaModel model(model_path);

    // Let memory stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    double idle_mem_mb = measureMemoryMB();
    std::cerr << "Scenario 1 - Idle memory: " << std::fixed << std::setprecision(0)
              << idle_mem_mb << " MB" << std::endl;

    // 3. Single Short Doc (9w) - Warmup + Measure
    std::cerr << "[3/6] Scenario 2 - Single short doc (9w): warmup + 20 runs..." << std::endl;
    std::string short_doc = getShortDoc();
    warmup(model, short_doc);
    auto short_latency = measureSingleDocLatency(model, short_doc);
    std::cerr << "Scenario 2 - Short doc P50: " << std::fixed << std::setprecision(1)
              << short_latency.p50 << " ms, P95: " << short_latency.p95
              << " ms, P99: " << short_latency.p99 << " ms" << std::endl;

    // 4. Single Long Doc (49w) - Warmup + Measure
    std::cerr << "[4/6] Scenario 3 - Single long doc (49w): warmup + 20 runs..." << std::endl;
    std::string long_doc = getLongDoc();
    warmup(model, long_doc);
    auto long_latency = measureSingleDocLatency(model, long_doc);
    std::cerr << "Scenario 3 - Long doc P50: " << std::fixed << std::setprecision(1)
              << long_latency.p50 << " ms, P95: " << long_latency.p95
              << " ms, P99: " << long_latency.p99 << " ms" << std::endl;

    // 5. Batch Short Docs (96x) - 20 seconds
    std::cerr << "[5/6] Scenario 4 - Batch short docs (96x): 20 seconds..." << std::endl;
    std::vector<std::string> short_docs;
    for (size_t i = 0; i < SHORT_DOCS.size(); i++) {
        short_docs.push_back(SHORT_DOCS[i]);
    }
    auto short_throughput = measureThroughput(model, short_docs, 96, 20);
    std::cerr << "Scenario 4 - Throughput: " << std::fixed << std::setprecision(1)
              << short_throughput.throughput << " emb/sec, Peak memory: "
              << std::fixed << std::setprecision(0) << short_throughput.peak_memory_mb
              << " MB" << std::endl;

    // 6. Batch Long Docs (96x) - 20 seconds
    std::cerr << "[6/6] Scenario 5 - Batch long docs (96x): 20 seconds..." << std::endl;
    std::vector<std::string> long_docs = {long_doc};
    auto long_throughput = measureThroughput(model, long_docs, 96, 20);
    std::cerr << "Scenario 5 - Throughput: " << std::fixed << std::setprecision(1)
              << long_throughput.throughput << " emb/sec, Peak memory: "
              << std::fixed << std::setprecision(0) << long_throughput.peak_memory_mb
              << " MB" << std::endl;

    // Format and print results
    metrics::OutputFormatter::printComprehensiveResults(
        platform,
        idle_mem_mb,
        short_latency,
        long_latency,
        short_throughput,
        long_throughput
    );
}

} // namespace benchmark
