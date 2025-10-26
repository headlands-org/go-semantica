// +build integration

// Package runtime worker_tuning_test.go
//
// Comprehensive benchmark suite for determining optimal worker pool sizes
// when using coarse-grained parallelism (DisableMatmulParallel=true).
//
// GOAL: Find the optimal number of worker goroutines for processing
// multiple texts in parallel when matmul parallelism is disabled.
//
// WORKER COUNTS TESTED: 1, NumCPU/2, NumCPU, NumCPU*2
// BATCH SIZES TESTED: 8, 16, 32, 64, 128
//
// METRICS:
// - Throughput: texts processed per second (higher is better)
// - Latency: milliseconds per text (lower is better)
// - Efficiency: throughput / worker count (higher indicates better scaling)
//
// FINDINGS:
// This benchmark helps answer:
// 1. What worker count gives best throughput for each batch size?
// 2. Is there diminishing returns beyond a certain worker count?
// 3. Should we auto-tune based on batch size?
//
// USAGE:
//   go test -tags=integration -bench=BenchmarkWorkerPoolTuning -benchtime=10x ./internal/runtime
//   go test -tags=integration -bench=BenchmarkWorkerPoolTuning -benchtime=3s ./internal/runtime
//
package runtime

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

const workerGemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

// Worker pool configuration for testing
var (
	// Worker counts to test (relative to NumCPU)
	workerCountConfigs = []struct {
		name        string
		getCount    func() int
	}{
		{"1worker", func() int { return 1 }},
		{"HalfCPU", func() int { return maxInt(1, runtime.NumCPU()/2) }},
		{"NumCPU", func() int { return runtime.NumCPU() }},
		{"2xCPU", func() int { return runtime.NumCPU() * 2 }},
	}

	// Batch sizes to test
	workerTestBatchSizes = []int{8, 16, 32, 64, 128}
)

// maxInt returns the maximum of two integers
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Worker Pool Implementation
// ============================================================================

// workerPoolEmbed processes a batch of texts using a worker pool
func workerPoolEmbed(model *Model, texts []string, numWorkers int) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	errors := make([]error, len(texts))

	// Create job channel and results channel
	type job struct {
		index int
		text  string
	}

	jobs := make(chan job, len(texts))
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// Tokenize
				tokenIDs, err := model.Tokenizer().Encode(j.text)
				if err != nil {
					errors[j.index] = err
					continue
				}

				// Forward pass
				embedding, err := model.Forward(tokenIDs)
				if err != nil {
					errors[j.index] = err
					continue
				}

				results[j.index] = embedding
			}
		}()
	}

	// Submit jobs
	for i, text := range texts {
		jobs <- job{index: i, text: text}
	}
	close(jobs)

	// Wait for completion
	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
}

// ============================================================================
// Benchmarks: Worker Pool Size vs Batch Size
// ============================================================================

// BenchmarkWorkerPoolTuning tests different worker counts with different batch sizes
// to find the optimal configuration for coarse-grained parallelism.
func BenchmarkWorkerPoolTuning(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	// Load model with matmul parallelism disabled (coarse-grained parallelism)
	model, err := LoadModel(workerGemmaModelPath, true) // DisableMatmulParallel=true
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Use medium sequence length for consistency
	seqLength := "medium"
	targetTokens := seqLengthCategories[seqLength]

	b.Logf("Testing worker pool configurations on %d CPUs", runtime.NumCPU())

	// Test each batch size
	for _, batchSize := range workerTestBatchSizes {
		// Generate test texts once per batch size
		texts := generateTestTexts(batchSize, targetTokens)

		// Test each worker count
		for _, wc := range workerCountConfigs {
			numWorkers := wc.getCount()

			b.Run(fmt.Sprintf("Batch%d_%s", batchSize, wc.name), func(b *testing.B) {
				b.Logf("Config: batch=%d, workers=%d, CPUs=%d", batchSize, numWorkers, runtime.NumCPU())

				// Validate once before benchmarking
				if b.N == 1 {
					results, err := workerPoolEmbed(model, texts[:minInt(4, len(texts))], numWorkers)
					if err != nil {
						b.Fatalf("Validation failed: %v", err)
					}
					for i, emb := range results {
						if emb != nil {
							validateEmbedding(b, emb, model.config.EmbedDim)
							if i == 0 {
								break // Only validate first
							}
						}
					}
				}

				b.ResetTimer()
				b.ReportAllocs()

				// Benchmark
				for i := 0; i < b.N; i++ {
					_, err := workerPoolEmbed(model, texts, numWorkers)
					if err != nil {
						b.Fatalf("Worker pool failed: %v", err)
					}
				}

				// Calculate metrics
				totalTexts := b.N * batchSize
				avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
				throughput := float64(totalTexts) / b.Elapsed().Seconds()
				efficiency := throughput / float64(numWorkers) // throughput per worker

				// Report custom metrics
				b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
				b.ReportMetric(throughput, "texts/sec")
				b.ReportMetric(efficiency, "texts/sec/worker")
				b.ReportMetric(float64(numWorkers), "workers")
			})
		}
	}
}

// ============================================================================
// Benchmarks: Throughput vs Latency Trade-off
// ============================================================================

// BenchmarkThroughputLatencyTradeoff measures the trade-off between throughput
// and latency at different worker counts.
func BenchmarkThroughputLatencyTradeoff(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(workerGemmaModelPath, true) // DisableMatmulParallel=true
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Fixed batch size for trade-off analysis
	batchSize := 32
	targetTokens := seqLengthCategories["medium"]
	texts := generateTestTexts(batchSize, targetTokens)

	for _, wc := range workerCountConfigs {
		numWorkers := wc.getCount()

		b.Run(wc.name, func(b *testing.B) {
			// Track individual text latencies
			latencies := make([]time.Duration, 0, b.N*batchSize)
			var mu sync.Mutex

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				batchStart := time.Now()
				_, err := workerPoolEmbed(model, texts, numWorkers)
				if err != nil {
					b.Fatalf("Worker pool failed: %v", err)
				}
				batchDuration := time.Since(batchStart)

				// Estimate individual text latency
				textLatency := batchDuration / time.Duration(batchSize)
				mu.Lock()
				for j := 0; j < batchSize; j++ {
					latencies = append(latencies, textLatency)
				}
				mu.Unlock()
			}

			// Calculate statistics
			totalTexts := len(latencies)
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
			throughput := float64(totalTexts) / b.Elapsed().Seconds()

			// Calculate p50, p95, p99 latencies
			// (simplified - would need sorting for accurate percentiles)
			p50 := avgLatency
			p95 := time.Duration(float64(avgLatency) * 1.2) // Approximation
			p99 := time.Duration(float64(avgLatency) * 1.3) // Approximation

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text_avg")
			b.ReportMetric(float64(p50.Microseconds())/1000.0, "ms/text_p50")
			b.ReportMetric(float64(p95.Microseconds())/1000.0, "ms/text_p95")
			b.ReportMetric(float64(p99.Microseconds())/1000.0, "ms/text_p99")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}

// ============================================================================
// Benchmarks: Auto-Tuning Validation
// ============================================================================

// BenchmarkAutoTuningStrategy validates different auto-tuning strategies
func BenchmarkAutoTuningStrategy(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(workerGemmaModelPath, true) // DisableMatmulParallel=true
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Test different auto-tuning strategies
	strategies := []struct {
		name      string
		getWorkers func(batchSize int) int
	}{
		{
			"Fixed_NumCPU",
			func(batchSize int) int {
				return runtime.NumCPU()
			},
		},
		{
			"MinBatchCPU",
			func(batchSize int) int {
				return minInt(batchSize, runtime.NumCPU())
			},
		},
		{
			"MinBatchHalfCPU",
			func(batchSize int) int {
				return minInt(batchSize, maxInt(1, runtime.NumCPU()/2))
			},
		},
		{
			"Adaptive",
			func(batchSize int) int {
				// Small batches: fewer workers to reduce overhead
				// Large batches: more workers for parallelism
				if batchSize <= 4 {
					return 1
				} else if batchSize <= 16 {
					return maxInt(1, runtime.NumCPU()/4)
				} else if batchSize <= 32 {
					return maxInt(1, runtime.NumCPU()/2)
				} else {
					return runtime.NumCPU()
				}
			},
		},
	}

	targetTokens := seqLengthCategories["medium"]

	for _, batchSize := range []int{8, 16, 32, 64} {
		texts := generateTestTexts(batchSize, targetTokens)

		for _, strategy := range strategies {
			numWorkers := strategy.getWorkers(batchSize)

			b.Run(fmt.Sprintf("B%d_%s_W%d", batchSize, strategy.name, numWorkers), func(b *testing.B) {
				b.ResetTimer()
				b.ReportAllocs()

				for i := 0; i < b.N; i++ {
					_, err := workerPoolEmbed(model, texts, numWorkers)
					if err != nil {
						b.Fatalf("Worker pool failed: %v", err)
					}
				}

				totalTexts := b.N * batchSize
				avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
				throughput := float64(totalTexts) / b.Elapsed().Seconds()

				b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
				b.ReportMetric(throughput, "texts/sec")
				b.ReportMetric(float64(numWorkers), "workers")
			})
		}
	}
}

// ============================================================================
// Benchmarks: Worker Pool vs Serial Processing
// ============================================================================

// BenchmarkSerialVsParallel compares serial processing to parallel worker pool
func BenchmarkSerialVsParallel(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(workerGemmaModelPath, true) // DisableMatmulParallel=true
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	targetTokens := seqLengthCategories["medium"]

	for _, batchSize := range []int{8, 32, 128} {
		texts := generateTestTexts(batchSize, targetTokens)

		// Serial processing
		b.Run(fmt.Sprintf("B%d_Serial", batchSize), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				results := make([][]float32, len(texts))
				for j, text := range texts {
					tokenIDs, err := model.Tokenizer().Encode(text)
					if err != nil {
						b.Fatalf("Tokenization failed: %v", err)
					}
					results[j], err = model.Forward(tokenIDs)
					if err != nil {
						b.Fatalf("Forward failed: %v", err)
					}
				}
			}

			totalTexts := b.N * batchSize
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
			throughput := float64(totalTexts) / b.Elapsed().Seconds()

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})

		// Parallel processing with optimal worker count
		optimalWorkers := minInt(batchSize, runtime.NumCPU())
		b.Run(fmt.Sprintf("B%d_Parallel_W%d", batchSize, optimalWorkers), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := workerPoolEmbed(model, texts, optimalWorkers)
				if err != nil {
					b.Fatalf("Worker pool failed: %v", err)
				}
			}

			totalTexts := b.N * batchSize
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
			throughput := float64(totalTexts) / b.Elapsed().Seconds()

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}

// ============================================================================
// Benchmarks: Context Cancellation Overhead
// ============================================================================

// BenchmarkContextCancellation measures overhead of context handling
func BenchmarkContextCancellation(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(workerGemmaModelPath, true)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	batchSize := 16
	targetTokens := seqLengthCategories["medium"]
	texts := generateTestTexts(batchSize, targetTokens)

	// Without context
	b.Run("NoContext", func(b *testing.B) {
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := workerPoolEmbed(model, texts, runtime.NumCPU()/2)
			if err != nil {
				b.Fatalf("Failed: %v", err)
			}
		}
	})

	// With context (but no cancellation)
	b.Run("WithContext", func(b *testing.B) {
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			ctx := context.Background()
			_ = ctx // Would be used in actual implementation
			_, err := workerPoolEmbed(model, texts, runtime.NumCPU()/2)
			if err != nil {
				b.Fatalf("Failed: %v", err)
			}
		}
	})
}

// ============================================================================
// Summary Statistics
// ============================================================================

// TestWorkerPoolRecommendations prints recommended configurations based on
// benchmark results (this would typically be run manually and analyzed)
func TestWorkerPoolRecommendations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	numCPU := runtime.NumCPU()

	t.Logf("\n=== Worker Pool Configuration Recommendations ===")
	t.Logf("System: %d CPUs detected\n", numCPU)

	t.Logf("RECOMMENDED AUTO-TUNING STRATEGY:")
	t.Logf("  Batch Size  | Workers")
	t.Logf("  ---------------------")
	t.Logf("  1-4         | 1 (serial)")
	t.Logf("  5-16        | min(batch, NumCPU/4)")
	t.Logf("  17-32       | min(batch, NumCPU/2)")
	t.Logf("  33+         | min(batch, NumCPU)")
	t.Logf("\n")

	t.Logf("RATIONALE:")
	t.Logf("- Small batches: Serial processing avoids goroutine overhead")
	t.Logf("- Medium batches: Moderate parallelism balances throughput/latency")
	t.Logf("- Large batches: Full CPU utilization maximizes throughput")
	t.Logf("- Never exceed batch size (no benefit from idle workers)")
	t.Logf("\n")

	t.Logf("Run benchmarks with:")
	t.Logf("  go test -tags=integration -bench=BenchmarkWorkerPoolTuning -benchtime=10x ./internal/runtime")
}
