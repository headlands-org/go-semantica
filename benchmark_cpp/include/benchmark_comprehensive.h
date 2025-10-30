#ifndef BENCHMARK_COMPREHENSIVE_H
#define BENCHMARK_COMPREHENSIVE_H

#include <string>

namespace benchmark {

/**
 * Run comprehensive benchmark suite with 5 scenarios matching Go implementation.
 *
 * Scenarios:
 * 1. Idle Memory - measures RSS after model load
 * 2. Single Short Doc (9w) - 20 iterations, reports P50/P95/P99 latency
 * 3. Single Long Doc (49w) - 20 iterations, reports P50/P95/P99 latency
 * 4. Batch Short Docs (96x) - 20s throughput test, reports emb/sec, peak memory, avg latency
 * 5. Batch Long Docs (96x) - 20s throughput test, reports emb/sec, peak memory, avg latency
 *
 * Output format: formatted table matching Go implementation
 * - Platform info header: CPU model, core count, OS/arch
 * - Table with columns: Scenario | Metric | Value | Unit
 *
 * @param model_path Path to the GGUF model file
 * @param threads llama.cpp CPU threads to use for measurements
 */
void runComprehensiveMode(const std::string& model_path, int threads);

} // namespace benchmark

#endif // BENCHMARK_COMPREHENSIVE_H
