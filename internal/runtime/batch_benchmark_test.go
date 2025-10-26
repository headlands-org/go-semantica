// +build integration

// Package runtime batch_benchmark_test.go
//
// Comprehensive benchmark suite for testing embedding model performance across
// various configurations:
//
// BATCH SIZES: 1, 4, 8, 16, 32, 64
// - Tests how throughput scales with batch size
// - Measures memory allocation per batch
//
// SEQUENCE LENGTHS: short (4 tokens), medium (64 tokens), long (256 tokens)
// - Tests performance impact of different input sizes
// - Validates handling of variable-length inputs
//
// THREAD COUNTS: 1, 2, 4, 8, 16
// - Tests multi-threaded throughput (with mutex protection)
// - Measures contention overhead (model is not thread-safe)
//
// METRICS REPORTED:
// - Latency: milliseconds per text (ms/text)
// - Throughput: texts processed per second (texts/sec)
// - Memory: allocations per operation (allocs/op, bytes/op)
// - Correctness: cosine similarity validation
//
// USAGE:
//   # Run all benchmarks
//   go test -tags=integration -bench=. ./internal/runtime
//
//   # Run specific benchmark suite
//   go test -tags=integration -bench=BenchmarkBatchSizes ./internal/runtime
//   go test -tags=integration -bench=BenchmarkSequenceLengths ./internal/runtime
//
//   # Control iterations
//   go test -tags=integration -bench=. -benchtime=10x ./internal/runtime
//   go test -tags=integration -bench=. -benchtime=5s ./internal/runtime
//
//   # With memory profiling
//   go test -tags=integration -bench=. -memprofile=mem.prof ./internal/runtime
//
// NOTE: Parallel benchmarks use mutex protection because the current model
// implementation is not thread-safe. This measures serialized throughput
// and contention overhead.

package runtime

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// Test configurations
var (
	// Batch sizes to test
	batchSizes = []int{1, 4, 8, 16, 32, 64}

	// Sequence length categories (in tokens)
	seqLengthCategories = map[string]int{
		"short":  4,   // ~1-2 words
		"medium": 64,  // ~40-50 words
		"long":   256, // ~170-200 words
	}

	// Thread counts to test
	threadCounts = []int{1, 2, 4, 8, 16}
)

// ============================================================================
// Helper Functions
// ============================================================================

// generateTestText creates text with approximately the target number of tokens
// Based on ~1.3 tokens per word average for English
func generateTestText(targetTokens int) string {
	if targetTokens <= 0 {
		return ""
	}

	// Use a variety of common words to ensure realistic tokenization
	words := []string{
		"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
		"hello", "world", "this", "is", "a", "test", "sentence", "for",
		"benchmarking", "performance", "with", "different", "sequence", "lengths",
		"embedding", "models", "natural", "language", "processing", "machine",
		"learning", "artificial", "intelligence", "neural", "networks", "deep",
		"transformer", "attention", "mechanism", "computer", "science", "data",
	}

	// Estimate words needed (accounting for tokenization overhead)
	wordsNeeded := int(float64(targetTokens) * 0.8) // Conservative estimate

	var builder strings.Builder
	for i := 0; i < wordsNeeded; i++ {
		if i > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(words[i%len(words)])
	}

	return builder.String()
}

// generateTestTexts creates a batch of test texts with the target token count
func generateTestTexts(batchSize, targetTokens int) []string {
	texts := make([]string, batchSize)
	baseText := generateTestText(targetTokens)
	for i := 0; i < batchSize; i++ {
		// Add slight variation to each text
		texts[i] = fmt.Sprintf("%s item %d", baseText, i)
	}
	return texts
}

// Note: cosineSimilarity is defined in embeddinggemma_test.go

// validateEmbedding checks if the embedding is valid and reasonable
func validateEmbedding(t testing.TB, emb []float32, expectedDim int) {
	if len(emb) != expectedDim {
		t.Errorf("Wrong embedding dimension: got %d, want %d", len(emb), expectedDim)
		return
	}

	// Check L2 norm is approximately 1.0 (normalized embeddings)
	var norm float32
	for _, v := range emb {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if math.Abs(float64(norm-1.0)) > 0.01 {
		t.Errorf("Embedding not normalized: L2 norm = %.6f, want ~1.0", norm)
	}

	// Check for NaN or Inf
	for i, v := range emb {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at index %d", i)
			return
		}
		if math.IsInf(float64(v), 0) {
			t.Errorf("Inf at index %d", i)
			return
		}
	}
}

// validateConsistency checks if two embeddings are similar (for reproducibility)
func validateConsistency(t testing.TB, emb1, emb2 []float32, minSimilarity float32) {
	sim := cosineSimilarity(emb1, emb2)
	if sim < minSimilarity {
		t.Errorf("Embeddings not consistent: cosine similarity = %.6f, want >= %.6f", sim, minSimilarity)
	}
}

// ============================================================================
// Benchmark Results Reporting
// ============================================================================

// BenchmarkResult holds metrics for a single benchmark run
type BenchmarkResult struct {
	BatchSize      int
	SeqLength      string
	ThreadCount    int
	TotalTexts     int
	TotalTime      time.Duration
	AvgLatency     time.Duration // Per text
	Throughput     float64       // Texts per second
	AllocsPerOp    uint64
	BytesPerOp     uint64
}

// Report prints a formatted benchmark result
func (br *BenchmarkResult) Report() string {
	return fmt.Sprintf(
		"Batch=%2d Seq=%-6s Threads=%2d │ Latency=%6.2fms/text │ Throughput=%7.1f texts/sec │ Allocs=%6d/op │ Mem=%8.1fKB/op",
		br.BatchSize,
		br.SeqLength,
		br.ThreadCount,
		float64(br.AvgLatency.Microseconds())/1000.0,
		br.Throughput,
		br.AllocsPerOp,
		float64(br.BytesPerOp)/1024.0,
	)
}

// ============================================================================
// Single-Threaded Benchmarks
// ============================================================================

// BenchmarkBatchSizes tests different batch sizes with a fixed sequence length
func BenchmarkBatchSizes(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Use medium sequence length for batch size testing
	seqLength := "medium"
	targetTokens := seqLengthCategories[seqLength]

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch%d_%s", batchSize, seqLength), func(b *testing.B) {
			texts := generateTestTexts(batchSize, targetTokens)

			// Pre-tokenize to measure only inference time
			tokenIDsBatch := make([][]int, batchSize)
			for i, text := range texts {
				tokenIDs, err := model.Tokenizer().Encode(text)
				if err != nil {
					b.Fatalf("Tokenization failed: %v", err)
				}
				tokenIDsBatch[i] = tokenIDs
			}

			// Validate one embedding for correctness
			if b.N == 1 {
				emb, err := model.Forward(tokenIDsBatch[0])
				if err != nil {
					b.Fatalf("Forward pass failed: %v", err)
				}
				validateEmbedding(b, emb, model.config.EmbedDim)
			}

			b.ResetTimer()
			b.ReportAllocs()

			// Run benchmark
			for i := 0; i < b.N; i++ {
				for j := 0; j < batchSize; j++ {
					_, err := model.Forward(tokenIDsBatch[j])
					if err != nil {
						b.Fatalf("Forward pass failed: %v", err)
					}
				}
			}

			// Calculate and report metrics
			totalTexts := b.N * batchSize
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
			throughput := float64(totalTexts) / b.Elapsed().Seconds()

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}

// BenchmarkSequenceLengths tests different sequence lengths with a fixed batch size
func BenchmarkSequenceLengths(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Use small batch size for sequence length testing
	batchSize := 4

	// Test in order: short, medium, long
	for _, seqLength := range []string{"short", "medium", "long"} {
		targetTokens := seqLengthCategories[seqLength]

		b.Run(fmt.Sprintf("Batch%d_%s", batchSize, seqLength), func(b *testing.B) {
			texts := generateTestTexts(batchSize, targetTokens)

			// Pre-tokenize
			tokenIDsBatch := make([][]int, batchSize)
			for i, text := range texts {
				tokenIDs, err := model.Tokenizer().Encode(text)
				if err != nil {
					b.Fatalf("Tokenization failed: %v", err)
				}
				tokenIDsBatch[i] = tokenIDs
				if i == 0 {
					b.Logf("Sequence %s: %d tokens (target: %d)", seqLength, len(tokenIDs), targetTokens)
				}
			}

			// Validate one embedding
			if b.N == 1 {
				emb, err := model.Forward(tokenIDsBatch[0])
				if err != nil {
					b.Fatalf("Forward pass failed: %v", err)
				}
				validateEmbedding(b, emb, model.config.EmbedDim)
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				for j := 0; j < batchSize; j++ {
					_, err := model.Forward(tokenIDsBatch[j])
					if err != nil {
						b.Fatalf("Forward pass failed: %v", err)
					}
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
// Multi-Threaded Benchmarks (CPU Saturation)
// ============================================================================

// BenchmarkParallelThroughput tests CPU saturation with parallel workers
// NOTE: Current model implementation is NOT thread-safe. This benchmark uses
// a mutex to serialize access and measure contention overhead.
func BenchmarkParallelThroughput(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Use medium sequence length for parallel testing
	seqLength := "medium"
	targetTokens := seqLengthCategories[seqLength]

	// Test with smaller thread counts since model isn't thread-safe
	parallelThreads := []int{1, 2, 4}

	for _, threadCount := range parallelThreads {
		b.Run(fmt.Sprintf("Threads%d_%s", threadCount, seqLength), func(b *testing.B) {
			// Pre-generate and tokenize test data
			texts := generateTestTexts(threadCount*4, targetTokens) // 4x for variety
			tokenIDsBatch := make([][]int, len(texts))
			for i, text := range texts {
				tokenIDs, err := model.Tokenizer().Encode(text)
				if err != nil {
					b.Fatalf("Tokenization failed: %v", err)
				}
				tokenIDsBatch[i] = tokenIDs
			}

			// Set thread count
			oldGOMAXPROCS := runtime.GOMAXPROCS(threadCount)
			defer runtime.GOMAXPROCS(oldGOMAXPROCS)

			b.ResetTimer()
			b.ReportAllocs()

			// Use mutex to protect non-thread-safe model
			var mu sync.Mutex

			// Use RunParallel for CPU saturation testing (with serialization)
			b.RunParallel(func(pb *testing.PB) {
				localOps := 0
				for pb.Next() {
					idx := localOps % len(tokenIDsBatch)

					// Model is not thread-safe, so we must serialize access
					mu.Lock()
					_, err := model.Forward(tokenIDsBatch[idx])
					mu.Unlock()

					if err != nil {
						b.Errorf("Forward pass failed: %v", err)
					}
					localOps++
				}
			})

			// Calculate throughput
			throughput := float64(b.N) / b.Elapsed().Seconds()
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(b.N))

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}

// ============================================================================
// Worker Pool Benchmarks
// ============================================================================

// BenchmarkWorkerPool tests throughput with a fixed-size worker pool
// NOTE: Uses mutex protection since model is not thread-safe
func BenchmarkWorkerPool(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	seqLength := "medium"
	targetTokens := seqLengthCategories[seqLength]

	// Limit workers since model isn't thread-safe
	for _, workers := range []int{1, 2, 4} {
		b.Run(fmt.Sprintf("Workers%d_%s", workers, seqLength), func(b *testing.B) {
			// Pre-generate test data
			texts := generateTestTexts(workers*4, targetTokens)
			tokenIDsBatch := make([][]int, len(texts))
			for i, text := range texts {
				tokenIDs, err := model.Tokenizer().Encode(text)
				if err != nil {
					b.Fatalf("Tokenization failed: %v", err)
				}
				tokenIDsBatch[i] = tokenIDs
			}

			b.ResetTimer()
			b.ReportAllocs()

			// Create worker pool with mutex protection
			jobs := make(chan int, b.N)
			results := make(chan error, b.N)
			var mu sync.Mutex // Protect non-thread-safe model

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for idx := range jobs {
						tokenIdx := idx % len(tokenIDsBatch)

						// Serialize model access
						mu.Lock()
						_, err := model.Forward(tokenIDsBatch[tokenIdx])
						mu.Unlock()

						results <- err
					}
				}()
			}

			// Send jobs
			go func() {
				for i := 0; i < b.N; i++ {
					jobs <- i
				}
				close(jobs)
			}()

			// Wait for completion
			go func() {
				wg.Wait()
				close(results)
			}()

			// Collect results
			errorCount := 0
			for err := range results {
				if err != nil {
					errorCount++
				}
			}

			if errorCount > 0 {
				b.Errorf("%d forward passes failed", errorCount)
			}

			throughput := float64(b.N) / b.Elapsed().Seconds()
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(b.N))

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}

// ============================================================================
// Comprehensive Matrix Benchmark
// ============================================================================

// BenchmarkComprehensiveMatrix runs a full matrix of configurations
func BenchmarkComprehensiveMatrix(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Test subset of configurations for comprehensive view
	testBatchSizes := []int{1, 8, 32}
	testSeqLengths := []string{"short", "medium", "long"}

	for _, batchSize := range testBatchSizes {
		for _, seqLength := range testSeqLengths {
			targetTokens := seqLengthCategories[seqLength]

			b.Run(fmt.Sprintf("B%d_%s", batchSize, seqLength), func(b *testing.B) {
				texts := generateTestTexts(batchSize, targetTokens)

				// Pre-tokenize
				tokenIDsBatch := make([][]int, batchSize)
				actualTokens := 0
				for i, text := range texts {
					tokenIDs, err := model.Tokenizer().Encode(text)
					if err != nil {
						b.Fatalf("Tokenization failed: %v", err)
					}
					tokenIDsBatch[i] = tokenIDs
					if i == 0 {
						actualTokens = len(tokenIDs)
					}
				}

				b.Logf("Config: batch=%d, seq=%s (target=%d, actual=%d tokens)",
					batchSize, seqLength, targetTokens, actualTokens)

				b.ResetTimer()
				b.ReportAllocs()

				for i := 0; i < b.N; i++ {
					for j := 0; j < batchSize; j++ {
						_, err := model.Forward(tokenIDsBatch[j])
						if err != nil {
							b.Fatalf("Forward pass failed: %v", err)
						}
					}
				}

				totalTexts := b.N * batchSize
				avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
				throughput := float64(totalTexts) / b.Elapsed().Seconds()

				b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
				b.ReportMetric(throughput, "texts/sec")
				b.ReportMetric(float64(actualTokens), "tokens")
			})
		}
	}
}

// ============================================================================
// Correctness Validation Benchmarks
// ============================================================================

// BenchmarkWithValidation runs benchmarks while validating output correctness
func BenchmarkWithValidation(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	text := "Hello world"
	tokenIDs, err := model.Tokenizer().Encode(text)
	if err != nil {
		b.Fatalf("Tokenization failed: %v", err)
	}

	// Generate reference embedding
	referenceEmb, err := model.Forward(tokenIDs)
	if err != nil {
		b.Fatalf("Reference forward pass failed: %v", err)
	}
	validateEmbedding(b, referenceEmb, model.config.EmbedDim)

	b.ResetTimer()
	b.ReportAllocs()

	minSimilarity := float32(0.9999) // Very high threshold for consistency

	for i := 0; i < b.N; i++ {
		emb, err := model.Forward(tokenIDs)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}

		// Validate every iteration
		validateEmbedding(b, emb, model.config.EmbedDim)
		validateConsistency(b, emb, referenceEmb, minSimilarity)
	}

	avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(b.N))
	b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
}

// ============================================================================
// End-to-End Benchmarks (Tokenization + Inference)
// ============================================================================

// BenchmarkEndToEndPipeline measures complete pipeline including tokenization
func BenchmarkEndToEndPipeline(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	testCases := []struct {
		name      string
		batchSize int
		seqLength string
	}{
		{"B1_short", 1, "short"},
		{"B8_medium", 8, "medium"},
		{"B32_long", 32, "long"},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			targetTokens := seqLengthCategories[tc.seqLength]
			texts := generateTestTexts(tc.batchSize, targetTokens)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				for j := 0; j < tc.batchSize; j++ {
					// Include tokenization in measurement
					tokenIDs, err := model.Tokenizer().Encode(texts[j])
					if err != nil {
						b.Fatalf("Tokenization failed: %v", err)
					}

					_, err = model.Forward(tokenIDs)
					if err != nil {
						b.Fatalf("Forward pass failed: %v", err)
					}
				}
			}

			totalTexts := b.N * tc.batchSize
			avgLatency := time.Duration(b.Elapsed().Nanoseconds() / int64(totalTexts))
			throughput := float64(totalTexts) / b.Elapsed().Seconds()

			b.ReportMetric(float64(avgLatency.Microseconds())/1000.0, "ms/text")
			b.ReportMetric(throughput, "texts/sec")
		})
	}
}
