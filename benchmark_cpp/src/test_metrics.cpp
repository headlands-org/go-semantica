#include "metrics.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace metrics;

void testLatencyTracker() {
    std::cout << "Testing LatencyTracker...\n";

    LatencyTracker tracker;

    // Add some sample measurements (in milliseconds)
    tracker.addMeasurementMs(10.5);
    tracker.addMeasurementMs(12.3);
    tracker.addMeasurementMs(9.8);
    tracker.addMeasurementMs(15.2);
    tracker.addMeasurementMs(11.1);
    tracker.addMeasurementMs(13.7);
    tracker.addMeasurementMs(10.2);
    tracker.addMeasurementMs(14.5);
    tracker.addMeasurementMs(11.8);
    tracker.addMeasurementMs(12.9);

    auto stats = tracker.calculate();

    std::cout << "  Count: " << stats.count << "\n";
    std::cout << "  Mean: " << stats.mean << " ms\n";
    std::cout << "  P50: " << stats.p50 << " ms\n";
    std::cout << "  P95: " << stats.p95 << " ms\n";
    std::cout << "  P99: " << stats.p99 << " ms\n";
    std::cout << "\n";
}

void testThroughputCounter() {
    std::cout << "Testing ThroughputCounter...\n";

    ThroughputCounter counter;
    counter.start();

    // Simulate some work
    for (int i = 0; i < 100; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        counter.increment();
    }

    counter.stop();

    std::cout << "  Duration: " << counter.getDurationSeconds() << " seconds\n";
    std::cout << "  Count: " << counter.getCount() << "\n";
    std::cout << "  Throughput: " << counter.getThroughput() << " items/sec\n";
    std::cout << "  Avg Latency: " << counter.getAvgLatencyMs() << " ms\n";
    std::cout << "\n";
}

void testMemoryStats() {
    std::cout << "Testing MemoryStats...\n";

    MemoryStats stats;
    double rss = stats.getRssMB();

    std::cout << "  Current RSS: " << rss << " MB\n";

    // Allocate some memory
    const size_t alloc_size = 10 * 1024 * 1024; // 10 MB
    char* buffer = new char[alloc_size];

    // Touch the memory to ensure it's allocated
    for (size_t i = 0; i < alloc_size; i += 4096) {
        buffer[i] = 0;
    }

    stats.snapshot();
    double new_rss = stats.getRssMB();
    std::cout << "  After 10MB allocation: " << new_rss << " MB\n";
    std::cout << "  Peak RSS: " << stats.getPeakMB() << " MB\n";

    delete[] buffer;
    std::cout << "\n";
}

void testOutputFormatter() {
    std::cout << "Testing OutputFormatter...\n";

    // Test batch benchmark output
    OutputFormatter::printBenchmarkResults(
        "batch",
        2.126906174,
        192,
        90.27,
        11.077636
    );

    // Test worker stats output
    OutputFormatter::printWorkerStats(8, 22, 28, 24.5);

    // Test memory stats output
    OutputFormatter::printMemoryStats(64.90, 2480.06, 42);

    std::cout << "\n";
}

void testComprehensiveOutput() {
    std::cout << "Testing Comprehensive Output Format...\n";

    OutputFormatter::PlatformInfo platform;
    platform.cpu = "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz";
    platform.cores = 12;
    platform.os = "linux";
    platform.arch = "amd64";

    OutputFormatter::LatencyStats short_latency = {17.2, 17.5, 18.3, 19.1};
    OutputFormatter::LatencyStats long_latency = {52.8, 53.2, 55.1, 56.3};

    OutputFormatter::ThroughputStats short_throughput = {
        90.3,   // throughput
        125.5,  // peak_memory_mb
        20.0,   // duration
        1806    // total_embeddings
    };

    OutputFormatter::ThroughputStats long_throughput = {
        29.7,   // throughput
        132.8,  // peak_memory_mb
        20.0,   // duration
        594     // total_embeddings
    };

    OutputFormatter::printComprehensiveResults(
        platform,
        118.0,  // idle_mem_mb
        short_latency,
        long_latency,
        short_throughput,
        long_throughput
    );
}

void testPlatformDetection() {
    std::cout << "Testing Platform Detection...\n";

    std::cout << "  CPU: " << detectCPU() << "\n";
    std::cout << "  Cores: " << detectCores() << "\n";
    std::cout << "  OS: " << detectOS() << "\n";
    std::cout << "  Arch: " << detectArch() << "\n";
    std::cout << "\n";
}

int main() {
    std::cout << "=== Metrics Module Test Suite ===\n\n";

    testPlatformDetection();
    testLatencyTracker();
    testThroughputCounter();
    testMemoryStats();
    testOutputFormatter();
    testComprehensiveOutput();

    std::cout << "=== All Tests Complete ===\n";

    return 0;
}
