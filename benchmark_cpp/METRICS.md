# Metrics Collection Module

The metrics module provides comprehensive performance measurement capabilities for the C++ benchmark, matching the Go benchmark's output format.

## Components

### 1. LatencyTracker

Stores individual timing measurements and calculates percentile statistics.

```cpp
#include "metrics.h"
using namespace metrics;

LatencyTracker tracker;

// Measure latency
auto start = std::chrono::high_resolution_clock::now();
// ... do work ...
auto end = std::chrono::high_resolution_clock::now();
tracker.addMeasurement(end - start);

// Or add measurements directly in milliseconds
tracker.addMeasurementMs(15.3);

// Calculate statistics
auto stats = tracker.calculate();
std::cout << "P50: " << stats.p50 << " ms\n";
std::cout << "P95: " << stats.p95 << " ms\n";
std::cout << "P99: " << stats.p99 << " ms\n";
```

### 2. ThroughputCounter

Tracks throughput and average latency over a time period.

```cpp
ThroughputCounter counter;
counter.start();

for (int i = 0; i < 100; i++) {
    // ... generate embedding ...
    counter.increment();
}

counter.stop();

std::cout << "Throughput: " << counter.getThroughput() << " items/sec\n";
std::cout << "Avg Latency: " << counter.getAvgLatencyMs() << " ms\n";
```

### 3. MemoryStats

Measures RSS (Resident Set Size) memory usage.

**Linux**: Reads from `/proc/self/status`
**macOS**: Uses `task_info` API
**Windows**: Not yet implemented

```cpp
MemoryStats stats;

// Get current memory usage
std::cout << "Current RSS: " << stats.getRssMB() << " MB\n";

// Track peak memory with periodic snapshots
for (int i = 0; i < iterations; i++) {
    // ... do work ...
    stats.snapshot();  // Update peak if current > peak
}

std::cout << "Peak RSS: " << stats.getPeakMB() << " MB\n";
```

### 4. OutputFormatter

Formats output to match Go benchmark style.

```cpp
// Format batch/isolated mode results
OutputFormatter::printBenchmarkResults(
    "batch",              // mode
    2.126906174,         // duration_seconds
    192,                 // total_embeddings
    90.27,               // throughput
    11.077636            // avg_latency_ms
);

// Format per-worker statistics
OutputFormatter::printWorkerStats(
    8,      // num_workers
    22,     // min_count
    28,     // max_count
    24.5    // avg_count
);

// Format memory statistics
OutputFormatter::printMemoryStats(
    64.90,    // heap_alloc_mb
    2480.06,  // total_alloc_mb
    42        // num_gc
);

// Format comprehensive 5-scenario benchmark results
OutputFormatter::PlatformInfo platform;
platform.cpu = detectCPU();
platform.cores = detectCores();
platform.os = detectOS();
platform.arch = detectArch();

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
    118.0,           // idle_mem_mb
    short_latency,
    long_latency,
    short_throughput,
    long_throughput
);
```

## Platform Detection Utilities

```cpp
std::string cpu = detectCPU();      // e.g., "Intel(R) Core(TM) i7-9750H"
int cores = detectCores();          // e.g., 12
std::string os = detectOS();        // "linux", "darwin", "windows"
std::string arch = detectArch();    // "amd64", "arm64", "386", "arm"
```

## Output Format

The output formatters produce output matching the Go benchmark:

### Batch/Isolated Mode
```
=== Benchmark Results ===
Mode: batch
Duration: 2.126906174s
Total embeddings: 192
Throughput: 90.27 embeddings/sec
Average latency: 11.077636ms per embedding

=== Per-Worker Statistics ===
Workers: 8
Min embeddings per worker: 22
Max embeddings per worker: 28
Avg embeddings per worker: 24.50

=== Memory Statistics ===
HeapAlloc: 64.90 MB
TotalAlloc: 2480.06 MB
NumGC: 42
```

### Comprehensive Mode (5-Scenario)
```
=== Benchmark Results ===

Platform: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 cores, linux/amd64

Scenario                        Metric              Value       Unit
------------------------------------------------------------------------
Idle Memory                     Heap Allocated      118         MB

Single Short Doc (9w)           P50 Latency         17.5        ms
                                P95 Latency         18.3        ms
                                P99 Latency         19.1        ms

Single Long Doc (49w)           P50 Latency         53.2        ms
                                P95 Latency         55.1        ms
                                P99 Latency         56.3        ms

Batch Short Docs (96x)          Throughput          90.3        emb/sec
                                Peak Memory         126         MB
                                Avg Latency         11.1        ms/emb

Batch Long Docs (96x)           Throughput          29.7        emb/sec
                                Peak Memory         133         MB
                                Avg Latency         33.7        ms/emb
```

## Building and Testing

Build the metrics test executable:
```bash
cd benchmark_cpp
make test_metrics
./build/test_metrics
```

The test executable validates all components and demonstrates the output format.

## Implementation Notes

### High-Resolution Timing
- Uses `std::chrono::high_resolution_clock` for microsecond precision
- Latency measurements stored internally in milliseconds
- Duration calculations use `std::chrono::duration<double, std::milli>`

### Percentile Calculation
- Measurements sorted on-demand (lazy sorting)
- Percentile indices calculated as `size * percentile` (matching Go implementation)
- Indices clamped to valid range to prevent overflow

### Memory Measurement
- **Linux**: Parses `VmRSS:` line from `/proc/self/status` (RSS in kB)
- **macOS**: Uses `task_info()` with `MACH_TASK_BASIC_INFO` to get `resident_size`
- **Overhead**: RSS includes all allocated memory, stack, heap, shared libraries
- **Peak tracking**: Requires periodic `snapshot()` calls during benchmark

### Thread Safety
- Classes are **NOT thread-safe** by design (matches Go benchmark)
- Use separate instances per thread if needed
- Memory snapshots can be called from background threads

## Future Enhancements

Potential improvements:
- [ ] Windows memory measurement support (`GetProcessMemoryInfo`)
- [ ] Thread-safe versions with mutexes (if parallel benchmarking needed)
- [ ] CSV export for plotting/analysis
- [ ] Histogram generation for latency distributions
- [ ] Configurable percentiles (P90, P99.9, etc.)
