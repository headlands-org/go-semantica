#ifndef METRICS_H
#define METRICS_H

#include <chrono>
#include <vector>
#include <string>
#include <cstdint>

namespace metrics {

// High-resolution time point and duration types
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::high_resolution_clock::duration;

// LatencyTracker stores individual timing measurements and calculates percentiles
class LatencyTracker {
public:
    LatencyTracker();

    // Add a single latency measurement
    void addMeasurement(Duration duration);

    // Add a latency measurement in milliseconds
    void addMeasurementMs(double milliseconds);

    // Calculate statistics (sorts measurements if not already sorted)
    struct Stats {
        double mean;     // Mean latency in milliseconds
        double p50;      // P50 percentile (median) in milliseconds
        double p95;      // P95 percentile in milliseconds
        double p99;      // P99 percentile in milliseconds
        size_t count;    // Number of measurements
    };

    Stats calculate();

    // Get raw measurements (in milliseconds)
    const std::vector<double>& getMeasurements() const { return measurements_; }

    // Clear all measurements
    void clear();

private:
    std::vector<double> measurements_;  // Stored in milliseconds
    bool sorted_;
};

// ThroughputCounter tracks start/end times and counts for throughput calculation
class ThroughputCounter {
public:
    ThroughputCounter();

    // Start timing
    void start();

    // Stop timing
    void stop();

    // Increment count (e.g., number of embeddings generated)
    void increment(int count = 1);

    // Get duration in seconds
    double getDurationSeconds() const;

    // Get count
    int getCount() const { return count_; }

    // Calculate throughput (items/second)
    double getThroughput() const;

    // Get average latency per item (milliseconds)
    double getAvgLatencyMs() const;

    // Get total CPU time consumed between start and stop (seconds)
    double getCpuSeconds() const;

    // Reset counter
    void reset();

private:
    TimePoint start_time_;
    TimePoint end_time_;
    int count_;
    bool started_;
    bool stopped_;
    double cpu_start_seconds_;
    double cpu_end_seconds_;
};

// MemoryStats measures RSS (Resident Set Size) memory usage
class MemoryStats {
public:
    MemoryStats();

    // Get current RSS in bytes
    uint64_t getRssBytes() const;

    // Get current RSS in megabytes
    double getRssMB() const;

    // Take a memory snapshot
    void snapshot();

    // Get peak RSS observed (requires periodic snapshots)
    double getPeakMB() const { return peak_mb_; }

private:
    uint64_t parseRssLinux() const;
    uint64_t parseRssMacOS() const;

    double peak_mb_;
};

// OutputFormatter provides formatted output matching Go benchmark style
class OutputFormatter {
public:
    // Format batch/isolated mode benchmark results
    static void printBenchmarkResults(
        const std::string& mode,
        double duration_seconds,
        double compute_seconds,
        int total_embeddings,
        double throughput,
        double avg_latency_ms,
        double avg_compute_ms
    );

    // Format per-worker statistics for isolated mode
    static void printWorkerStats(
        int num_workers,
        int min_count,
        int max_count,
        double avg_count
    );

    // Format memory statistics
    static void printMemoryStats(
        double heap_alloc_mb,
        double total_alloc_mb,
        int num_gc
    );

    // Format comprehensive benchmark results (5-scenario)
    struct PlatformInfo {
        std::string cpu;
        int cores;
        std::string os;
        std::string arch;
    };

    struct LatencyStats {
        double mean;
        double p50;
        double p95;
        double p99;
        double compute_mean;
        double compute_p50;
        double compute_p95;
        double compute_p99;
    };

    struct ThroughputStats {
        double throughput;
        double peak_memory_mb;
        double duration;
        int total_embeddings;
        double compute_seconds;
        double compute_per_ms;
    };

    static void printComprehensiveResults(
        const PlatformInfo& platform,
        double idle_mem_mb,
        const LatencyStats& short_latency,
        const LatencyStats& long_latency,
        const LatencyStats& extra_long_latency,
        const ThroughputStats& short_throughput,
        const ThroughputStats& long_throughput
    );
};

// Utility functions for platform detection
std::string detectCPU();
int detectCores();
std::string detectOS();
std::string detectArch();

// CPU time helper (seconds of user+system time consumed by current process)
double getProcessCpuTimeSeconds();

} // namespace metrics

#endif // METRICS_H
