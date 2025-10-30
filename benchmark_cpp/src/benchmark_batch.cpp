#include "benchmark_batch.h"
#include "model.h"
#include "test_data.h"
#include "metrics.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

namespace benchmark {

int runBatchMode(const std::string& modelPath, int duration, int batchSize, int threads) {
    int thread_count = threads > 0 ? threads : 1;
    try {
        std::cout << "Loading model from: " << modelPath << std::endl;

        // Load model once (shared across all iterations)
        embedding::LlamaModel model(modelPath, /*n_ctx=*/0, /*n_batch=*/16384, thread_count);

        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "  Embedding dimension: " << model.getEmbeddingDim() << std::endl;
        std::cout << "  Context size: " << model.getContextSize() << std::endl;
        std::cout << std::endl;

        std::cout << "Running batch mode benchmark for " << duration << " seconds..." << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << std::endl;

        // Initialize metrics
        metrics::ThroughputCounter throughputCounter;
        metrics::MemoryStats memStats;

        // Random number generator for selecting texts
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, getTestTextsSize() - 1);

        // Start timing
        throughputCounter.start();
        auto startTime = std::chrono::high_resolution_clock::now();
        auto deadline = startTime + std::chrono::seconds(duration);

        int totalEmbeddings = 0;

        // Run benchmark loop until deadline is reached
        while (true) {
            // Check if we've reached the deadline
            auto now = std::chrono::high_resolution_clock::now();
            if (now >= deadline) {
                break;
            }

            // Select batchSize random texts from the corpus
            std::vector<std::string> selectedTexts;
            selectedTexts.reserve(batchSize);
            for (int i = 0; i < batchSize; ++i) {
                size_t idx = dis(gen);
                selectedTexts.push_back(TEST_TEXTS[idx]);
            }

            // Generate embeddings for the batch
            try {
                auto embeddings = model.generateEmbeddingsBatch(selectedTexts);
                totalEmbeddings += embeddings.size();
                throughputCounter.increment(embeddings.size());
            } catch (const std::exception& e) {
                std::cerr << "Warning: embedding failed: " << e.what() << std::endl;
                continue;
            }
        }

        // Stop timing
        throughputCounter.stop();

        // Sample memory at end of run
        memStats.snapshot();

        // Calculate metrics
        double durationSeconds = throughputCounter.getDurationSeconds();
        double throughput = throughputCounter.getThroughput();
        double avgLatencyMs = throughputCounter.getAvgLatencyMs();
        double computeSeconds = throughputCounter.getCpuSeconds();
        double avgComputeMs = 0.0;
        if (totalEmbeddings > 0 && computeSeconds > 0.0) {
            avgComputeMs = (computeSeconds * 1000.0) / totalEmbeddings;
        }
        double rssMB = memStats.getRssMB();

        // Print results in Go format
        std::cout << std::endl;
        std::cout << "=== Benchmark Results ===" << std::endl;
        std::cout << "Mode: batch" << std::endl;
        std::cout << "Duration: " << std::fixed << std::setprecision(2)
                  << durationSeconds << "s" << std::endl;
        std::cout << "Total embeddings: " << totalEmbeddings << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                  << throughput << " embeddings/sec" << std::endl;
        std::cout << "Average latency: " << std::fixed << std::setprecision(2)
                  << avgLatencyMs << "ms per embedding" << std::endl;
        std::cout << "Compute time: " << std::fixed << std::setprecision(3)
                  << computeSeconds << "s (" << std::fixed << std::setprecision(3)
                  << avgComputeMs << "ms per embedding)" << std::endl;

        // Print memory statistics
        std::cout << std::endl;
        std::cout << "=== Memory Statistics ===" << std::endl;
        std::cout << "RSS: " << std::fixed << std::setprecision(2)
                  << rssMB << " MB" << std::endl;

        return 0;

    } catch (const embedding::ModelError& e) {
        std::cerr << "Model error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace benchmark
