// +build integration

// benchmark-comparison: Comprehensive benchmark comparing nested vs coarse-grained parallelism
//
// This tool runs systematic benchmarks comparing two parallelism strategies:
// 1. Baseline (Current): Nested parallelism with fine-grained matmul parallelism
// 2. Optimized: Coarse-grained parallelism with serial matmul
//
// Results are exported to CSV for analysis.
//
// Usage:
//   go run -tags=integration ./cmd/benchmark-comparison

package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

const (
	modelPath       = "model/embeddinggemma-300m-Q8_0.gguf"
	outputCSV       = "docs/BENCHMARK_RESULTS.csv"
	numIterations   = 5  // Reduced from 10 for faster execution
	warmupIters     = 1  // Reduced from 2
	mediumTextWords = 50 // For "medium" length texts
)

// BenchmarkConfig represents a configuration to test
type BenchmarkConfig struct {
	Name                  string
	DisableMatmulParallel bool
	Description           string
}

// BenchmarkResult holds all metrics for a single benchmark run
type BenchmarkResult struct {
	ConfigName      string
	BatchSize       int
	WorkerCount     int
	LatencyP50Ms    float64
	LatencyP95Ms    float64
	LatencyP99Ms    float64
	ThroughputTexts float64
	AllocsPerOp     uint64
	BytesPerOp      uint64
	CPUPercent      float64
}

// generateTestText creates a text with approximately N words
func generateTestText(numWords int) string {
	words := []string{
		"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
		"hello", "world", "this", "is", "a", "test", "sentence", "for",
		"benchmarking", "performance", "with", "different", "sequence", "lengths",
		"embedding", "models", "natural", "language", "processing", "machine",
		"learning", "artificial", "intelligence", "neural", "networks", "deep",
		"transformer", "attention", "mechanism", "computer", "science", "data",
	}

	var builder strings.Builder
	for i := 0; i < numWords; i++ {
		if i > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(words[i%len(words)])
	}
	return builder.String()
}

// runBenchmarkIteration runs a single benchmark iteration and returns latencies
func runBenchmarkIteration(rt ggufembed.Runtime, texts []string, batchSize int) ([]time.Duration, error) {
	latencies := make([]time.Duration, 0, batchSize)

	// Process each text individually and measure latency
	for _, text := range texts {
		start := time.Now()
		_, err := rt.Embed(context.Background(), []string{text})
		if err != nil {
			return nil, err
		}
		latencies = append(latencies, time.Since(start))
	}

	return latencies, nil
}

// calculatePercentile calculates the Nth percentile from sorted durations
func calculatePercentile(sortedDurations []time.Duration, percentile float64) time.Duration {
	if len(sortedDurations) == 0 {
		return 0
	}
	index := int(float64(len(sortedDurations)-1) * percentile / 100.0)
	if index < 0 {
		index = 0
	}
	if index >= len(sortedDurations) {
		index = len(sortedDurations) - 1
	}
	return sortedDurations[index]
}

// runBenchmark runs a complete benchmark for a specific configuration
func runBenchmark(config BenchmarkConfig, batchSize, workerCount int) (*BenchmarkResult, error) {
	log.Printf("Running: %s | Batch=%d | Workers=%d", config.Name, batchSize, workerCount)

	// Create runtime with appropriate options
	var opts []ggufembed.Option
	opts = append(opts, ggufembed.WithDisableMatmulParallel(config.DisableMatmulParallel))
	if workerCount > 0 {
		opts = append(opts, ggufembed.WithThreads(workerCount))
	}

	rt, err := ggufembed.Open(modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to open runtime: %w", err)
	}
	defer rt.Close()

	// Generate test texts
	texts := make([]string, batchSize)
	baseText := generateTestText(mediumTextWords)
	for i := 0; i < batchSize; i++ {
		texts[i] = fmt.Sprintf("%s item %d", baseText, i)
	}

	// Warmup
	for i := 0; i < warmupIters; i++ {
		_, err := rt.Embed(context.Background(), texts[:min(4, len(texts))])
		if err != nil {
			return nil, fmt.Errorf("warmup failed: %w", err)
		}
	}

	// Run benchmark iterations
	allLatencies := make([]time.Duration, 0, numIterations*batchSize)
	var totalAllocsBefore runtime.MemStats
	runtime.ReadMemStats(&totalAllocsBefore)

	startTime := time.Now()

	for iter := 0; iter < numIterations; iter++ {
		latencies, err := runBenchmarkIteration(rt, texts, batchSize)
		if err != nil {
			return nil, fmt.Errorf("iteration %d failed: %w", iter, err)
		}
		allLatencies = append(allLatencies, latencies...)
	}

	totalDuration := time.Since(startTime)

	var totalAllocsAfter runtime.MemStats
	runtime.ReadMemStats(&totalAllocsAfter)

	// Calculate statistics
	sort.Slice(allLatencies, func(i, j int) bool {
		return allLatencies[i] < allLatencies[j]
	})

	p50 := calculatePercentile(allLatencies, 50)
	p95 := calculatePercentile(allLatencies, 95)
	p99 := calculatePercentile(allLatencies, 99)

	totalTexts := numIterations * batchSize
	throughput := float64(totalTexts) / totalDuration.Seconds()

	allocsPerOp := (totalAllocsAfter.Mallocs - totalAllocsBefore.Mallocs) / uint64(totalTexts)
	bytesPerOp := (totalAllocsAfter.TotalAlloc - totalAllocsBefore.TotalAlloc) / uint64(totalTexts)

	// Estimate CPU usage (rough approximation)
	cpuPercent := 0.0 // Would need CPU profiling for accurate measurement

	result := &BenchmarkResult{
		ConfigName:      config.Name,
		BatchSize:       batchSize,
		WorkerCount:     workerCount,
		LatencyP50Ms:    float64(p50.Microseconds()) / 1000.0,
		LatencyP95Ms:    float64(p95.Microseconds()) / 1000.0,
		LatencyP99Ms:    float64(p99.Microseconds()) / 1000.0,
		ThroughputTexts: throughput,
		AllocsPerOp:     allocsPerOp,
		BytesPerOp:      bytesPerOp,
		CPUPercent:      cpuPercent,
	}

	log.Printf("  ✓ P50=%.2fms | P95=%.2fms | P99=%.2fms | Throughput=%.1f texts/sec",
		result.LatencyP50Ms, result.LatencyP95Ms, result.LatencyP99Ms, result.ThroughputTexts)

	return result, nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// writeResultsToCSV writes all benchmark results to a CSV file
func writeResultsToCSV(results []*BenchmarkResult, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Write header
	header := []string{
		"Configuration",
		"Batch Size",
		"Worker Count",
		"Latency P50 (ms)",
		"Latency P95 (ms)",
		"Latency P99 (ms)",
		"Throughput (texts/sec)",
		"Allocs/op",
		"Bytes/op",
		"CPU %",
	}
	if err := w.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write results
	for _, r := range results {
		row := []string{
			r.ConfigName,
			fmt.Sprintf("%d", r.BatchSize),
			fmt.Sprintf("%d", r.WorkerCount),
			fmt.Sprintf("%.2f", r.LatencyP50Ms),
			fmt.Sprintf("%.2f", r.LatencyP95Ms),
			fmt.Sprintf("%.2f", r.LatencyP99Ms),
			fmt.Sprintf("%.2f", r.ThroughputTexts),
			fmt.Sprintf("%d", r.AllocsPerOp),
			fmt.Sprintf("%d", r.BytesPerOp),
			fmt.Sprintf("%.1f", r.CPUPercent),
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("failed to write row: %w", err)
		}
	}

	return nil
}

func main() {
	log.Printf("Starting comprehensive benchmark comparison")
	log.Printf("Model: %s", modelPath)
	log.Printf("Output: %s", outputCSV)
	log.Printf("Iterations per config: %d", numIterations)
	log.Printf("System CPUs: %d", runtime.NumCPU())
	log.Printf("")

	// Define configurations to test
	configs := []BenchmarkConfig{
		{
			Name:                  "Baseline-Nested",
			DisableMatmulParallel: false,
			Description:           "Current: Nested parallelism with fine-grained matmul",
		},
		{
			Name:                  "Optimized-Coarse",
			DisableMatmulParallel: true,
			Description:           "Optimized: Coarse-grained parallelism with serial matmul",
		},
	}

	// Define test parameters (focused on key configurations)
	batchSizes := []int{1, 8, 16, 32, 64, 128}
	workerCounts := []int{8, 16} // Focus on most relevant worker counts

	// Collect all results
	var results []*BenchmarkResult
	var mu sync.Mutex

	// Run benchmarks
	totalBenchmarks := len(configs) * len(batchSizes) * len(workerCounts)
	currentBenchmark := 0

	for _, config := range configs {
		log.Printf("\n=== Configuration: %s ===", config.Name)
		log.Printf("Description: %s", config.Description)
		log.Printf("")

		for _, batchSize := range batchSizes {
			for _, workerCount := range workerCounts {
				currentBenchmark++
				log.Printf("[%d/%d] ", currentBenchmark, totalBenchmarks)

				result, err := runBenchmark(config, batchSize, workerCount)
				if err != nil {
					log.Printf("  ✗ ERROR: %v", err)
					continue
				}

				mu.Lock()
				results = append(results, result)
				mu.Unlock()
			}
		}
	}

	// Write results to CSV
	log.Printf("\n=== Writing results to %s ===", outputCSV)
	if err := writeResultsToCSV(results, outputCSV); err != nil {
		log.Fatalf("Failed to write CSV: %v", err)
	}

	// Print summary
	log.Printf("\n=== Summary ===")
	log.Printf("Total benchmarks completed: %d", len(results))
	log.Printf("Results exported to: %s", outputCSV)

	// Print quick comparison for key metrics
	log.Printf("\n=== Quick Comparison (Batch=32, Workers=8) ===")
	for _, config := range configs {
		for _, r := range results {
			if r.ConfigName == config.Name && r.BatchSize == 32 && r.WorkerCount == 8 {
				log.Printf("%s:", r.ConfigName)
				log.Printf("  P50 Latency:  %.2f ms", r.LatencyP50Ms)
				log.Printf("  P95 Latency:  %.2f ms", r.LatencyP95Ms)
				log.Printf("  Throughput:   %.1f texts/sec", r.ThroughputTexts)
				log.Printf("  Allocs/op:    %d", r.AllocsPerOp)
			}
		}
	}

	log.Printf("\nBenchmark comparison completed successfully!")
}
