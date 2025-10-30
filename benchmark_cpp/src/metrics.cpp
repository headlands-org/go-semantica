#include "metrics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thread>
#include <ctime>

#ifdef __linux__
#include <unistd.h>
#include <fstream>
#elif __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

namespace metrics {

double getProcessCpuTimeSeconds() {
    clock_t ticks = std::clock();
    if (ticks == static_cast<clock_t>(-1)) {
        return 0.0;
    }
    return static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

// ============================================================================
// LatencyTracker Implementation
// ============================================================================

LatencyTracker::LatencyTracker() : sorted_(false) {}

void LatencyTracker::addMeasurement(Duration duration) {
    double ms = std::chrono::duration<double, std::milli>(duration).count();
    measurements_.push_back(ms);
    sorted_ = false;
}

void LatencyTracker::addMeasurementMs(double milliseconds) {
    measurements_.push_back(milliseconds);
    sorted_ = false;
}

LatencyTracker::Stats LatencyTracker::calculate() {
    Stats stats = {0.0, 0.0, 0.0, 0.0, measurements_.size()};

    if (measurements_.empty()) {
        return stats;
    }

    // Sort measurements if not already sorted
    if (!sorted_) {
        std::sort(measurements_.begin(), measurements_.end());
        sorted_ = true;
    }

    // Calculate mean
    double sum = std::accumulate(measurements_.begin(), measurements_.end(), 0.0);
    stats.mean = sum / measurements_.size();

    // Calculate percentiles
    size_t n = measurements_.size();
    size_t idx50 = static_cast<size_t>(n * 0.50);
    size_t idx95 = static_cast<size_t>(n * 0.95);
    size_t idx99 = static_cast<size_t>(n * 0.99);

    // Ensure indices are within bounds
    if (idx50 >= n) idx50 = n - 1;
    if (idx95 >= n) idx95 = n - 1;
    if (idx99 >= n) idx99 = n - 1;

    stats.p50 = measurements_[idx50];
    stats.p95 = measurements_[idx95];
    stats.p99 = measurements_[idx99];

    return stats;
}

void LatencyTracker::clear() {
    measurements_.clear();
    sorted_ = false;
}

// ============================================================================
// ThroughputCounter Implementation
// ============================================================================

ThroughputCounter::ThroughputCounter()
    : count_(0), started_(false), stopped_(false), cpu_start_seconds_(0.0), cpu_end_seconds_(0.0) {}

void ThroughputCounter::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    started_ = true;
    stopped_ = false;
    cpu_start_seconds_ = getProcessCpuTimeSeconds();
    cpu_end_seconds_ = cpu_start_seconds_;
}

void ThroughputCounter::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    stopped_ = true;
    cpu_end_seconds_ = getProcessCpuTimeSeconds();
}

void ThroughputCounter::increment(int count) {
    count_ += count;
}

double ThroughputCounter::getDurationSeconds() const {
    if (!started_) {
        return 0.0;
    }

    TimePoint end = stopped_ ? end_time_ : std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
    return duration.count() / 1000000.0;
}

double ThroughputCounter::getThroughput() const {
    double duration = getDurationSeconds();
    if (duration <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(count_) / duration;
}

double ThroughputCounter::getAvgLatencyMs() const {
    if (count_ == 0) {
        return 0.0;
    }
    double duration = getDurationSeconds();
    return (duration * 1000.0) / count_;
}

double ThroughputCounter::getCpuSeconds() const {
    if (!started_) {
        return 0.0;
    }
    double end = stopped_ ? cpu_end_seconds_ : getProcessCpuTimeSeconds();
    double diff = end - cpu_start_seconds_;
    if (diff < 0.0) {
        return 0.0;
    }
    return diff;
}

void ThroughputCounter::reset() {
    count_ = 0;
    started_ = false;
    stopped_ = false;
    cpu_start_seconds_ = 0.0;
    cpu_end_seconds_ = 0.0;
}

// ============================================================================
// MemoryStats Implementation
// ============================================================================

MemoryStats::MemoryStats() : peak_mb_(0.0) {}

uint64_t MemoryStats::getRssBytes() const {
#ifdef __linux__
    return parseRssLinux();
#elif __APPLE__
    return parseRssMacOS();
#else
    return 0;
#endif
}

double MemoryStats::getRssMB() const {
    return getRssBytes() / (1024.0 * 1024.0);
}

void MemoryStats::snapshot() {
    double current_mb = getRssMB();
    if (current_mb > peak_mb_) {
        peak_mb_ = current_mb;
    }
}

uint64_t MemoryStats::parseRssLinux() const {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return 0;
    }

    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // Format: "VmRSS:    12345 kB"
            std::istringstream iss(line.substr(6));
            uint64_t rss_kb;
            iss >> rss_kb;
            return rss_kb * 1024; // Convert to bytes
        }
    }
#endif
    return 0;
}

uint64_t MemoryStats::parseRssMacOS() const {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

// ============================================================================
// OutputFormatter Implementation
// ============================================================================

void OutputFormatter::printBenchmarkResults(
    const std::string& mode,
    double duration_seconds,
    double compute_seconds,
    int total_embeddings,
    double throughput,
    double avg_latency_ms,
    double avg_compute_ms)
{
    std::cerr << "\n=== Benchmark Results ===\n";
    std::cerr << "Mode: " << mode << "\n";
    std::cerr << std::fixed << std::setprecision(9);
    std::cerr << "Duration: " << duration_seconds << "s\n";
    std::cerr << "Compute time: " << compute_seconds << "s\n";
    std::cerr << "Total embeddings: " << total_embeddings << "\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "Throughput: " << throughput << " embeddings/sec\n";
    std::cerr << std::fixed << std::setprecision(6);
    std::cerr << "Average latency: " << avg_latency_ms << "ms per embedding\n";
    std::cerr << "Average compute: " << avg_compute_ms << "ms per embedding\n";
}

void OutputFormatter::printWorkerStats(
    int num_workers,
    int min_count,
    int max_count,
    double avg_count)
{
    std::cerr << "\n=== Per-Worker Statistics ===\n";
    std::cerr << "Workers: " << num_workers << "\n";
    std::cerr << "Min embeddings per worker: " << min_count << "\n";
    std::cerr << "Max embeddings per worker: " << max_count << "\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "Avg embeddings per worker: " << avg_count << "\n";
}

void OutputFormatter::printMemoryStats(
    double heap_alloc_mb,
    double total_alloc_mb,
    int num_gc)
{
    std::cerr << "\n=== Memory Statistics ===\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "HeapAlloc: " << heap_alloc_mb << " MB\n";
    std::cerr << "TotalAlloc: " << total_alloc_mb << " MB\n";
    std::cerr << "NumGC: " << num_gc << "\n";
}

void OutputFormatter::printComprehensiveResults(
    const PlatformInfo& platform,
    double idle_mem_mb,
    const LatencyStats& short_latency,
    const LatencyStats& long_latency,
    const LatencyStats& extra_long_latency,
    const ThroughputStats& short_throughput,
    const ThroughputStats& long_throughput)
{
    std::cout << "\n=== Benchmark Results ===\n\n";

    std::cout << "Platform: " << platform.cpu << ", " << platform.cores
              << " cores, " << platform.os << "/" << platform.arch << "\n\n";

    // Column headers
    std::cout << std::left << std::setw(32) << "Scenario"
              << std::setw(20) << "Metric"
              << std::setw(12) << "Value"
              << "Unit\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Scenario 1: Idle Memory
    std::cout << std::left << std::setw(32) << "Idle Memory"
              << std::setw(20) << "Heap Allocated"
              << std::setw(12) << std::fixed << std::setprecision(0) << idle_mem_mb
              << "MB\n";
    std::cout << "\n";

    // Scenario 2: Single Short Doc (9w)
    std::cout << std::left << std::setw(32) << "Single Short Doc (9w)"
              << std::setw(20) << "P50 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P95 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P99 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.p99
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P50"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.compute_p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P95"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.compute_p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P99"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_latency.compute_p99
              << "ms\n";
    std::cout << "\n";

    // Scenario 3: Single Long Doc (49w)
    std::cout << std::left << std::setw(32) << "Single Long Doc (49w)"
              << std::setw(20) << "P50 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P95 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P99 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.p99
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P50"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.compute_p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P95"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.compute_p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P99"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_latency.compute_p99
              << "ms\n";
    std::cout << "\n";

    // Scenario 4: Single Extra-Long Doc (~400w)
    std::cout << std::left << std::setw(32) << "Single Extra-Long Doc (~400w)"
              << std::setw(20) << "P50 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P95 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "P99 Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.p99
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P50"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.compute_p50
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P95"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.compute_p95
              << "ms\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU P99"
              << std::setw(12) << std::fixed << std::setprecision(1) << extra_long_latency.compute_p99
              << "ms\n";
    std::cout << "\n";

    // Scenario 5: Batch Short Docs (96x)
    double avg_latency_short = 1000.0 / short_throughput.throughput; // Convert to ms/emb
    std::cout << std::left << std::setw(32) << "Batch Short Docs (96x)"
              << std::setw(20) << "Throughput"
              << std::setw(12) << std::fixed << std::setprecision(1) << short_throughput.throughput
              << "emb/sec\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "Peak Memory"
              << std::setw(12) << std::fixed << std::setprecision(0) << short_throughput.peak_memory_mb
              << "MB\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "Avg Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << avg_latency_short
              << "ms/emb\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU Time"
              << std::setw(12) << std::fixed << std::setprecision(3) << short_throughput.compute_seconds
              << "s\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU per Embedding"
              << std::setw(12) << std::fixed << std::setprecision(3) << short_throughput.compute_per_ms
              << "ms/emb\n";
    std::cout << "\n";

    // Scenario 6: Batch Long Docs (96x)
    double avg_latency_long = 1000.0 / long_throughput.throughput; // Convert to ms/emb
    std::cout << std::left << std::setw(32) << "Batch Long Docs (96x)"
              << std::setw(20) << "Throughput"
              << std::setw(12) << std::fixed << std::setprecision(1) << long_throughput.throughput
              << "emb/sec\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "Peak Memory"
              << std::setw(12) << std::fixed << std::setprecision(0) << long_throughput.peak_memory_mb
              << "MB\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "Avg Latency"
              << std::setw(12) << std::fixed << std::setprecision(1) << avg_latency_long
              << "ms/emb\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU Time"
              << std::setw(12) << std::fixed << std::setprecision(3) << long_throughput.compute_seconds
              << "s\n";
    std::cout << std::left << std::setw(32) << ""
              << std::setw(20) << "CPU per Embedding"
              << std::setw(12) << std::fixed << std::setprecision(3) << long_throughput.compute_per_ms
              << "ms/emb\n";
}

// ============================================================================
// Platform Detection Utilities
// ============================================================================

std::string detectCPU() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return "Unknown";
    }

    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.compare(0, 10, "model name") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string model = line.substr(colon + 1);
                // Trim leading whitespace
                size_t start = model.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    return model.substr(start);
                }
            }
        }
    }
#elif __APPLE__
    char buffer[256];
    size_t size = sizeof(buffer);

    // Try to get CPU brand string
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
        return std::string(buffer);
    }

    // Fallback to architecture detection
#ifdef __arm64__
    return "Apple Silicon";
#else
    return "Intel";
#endif
#endif
    return "Unknown";
}

int detectCores() {
    return std::thread::hardware_concurrency();
}

std::string detectOS() {
#ifdef __linux__
    return "linux";
#elif __APPLE__
    return "darwin";
#elif _WIN32
    return "windows";
#else
    return "unknown";
#endif
}

std::string detectArch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "amd64";
#elif defined(__i386__) || defined(_M_IX86)
    return "386";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

} // namespace metrics
