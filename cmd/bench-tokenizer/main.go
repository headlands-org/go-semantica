// Tokenization performance benchmark
//
// Characterizes tokenizer speed across variable-size documents.
// Mirrors the benchmarking approach from rs-semantica.
//
// Usage:
//
//	GO_SEMANTICA_MODEL=/path/to/model.gguf go run ./cmd/bench-tokenizer
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/headlands-org/go-semantica/internal/runtime"
)

func main() {
	modelPath := os.Getenv("GO_SEMANTICA_MODEL")
	if modelPath == "" {
		modelPath = "/home/lth/Downloads/embeddinggemma-300m-Q8_0.gguf"
	}

	fmt.Println("=== Tokenization Performance Benchmark ===")
	fmt.Printf("Model: %s\n\n", modelPath)

	// Load model (which includes tokenizer)
	loadStart := time.Now()
	model, err := runtime.LoadModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	tokenizer := model.Tokenizer()
	fmt.Printf("Tokenizer load: %v\n\n", time.Since(loadStart))

	// Test cases: varying document sizes
	testCases := []struct {
		name string
		text string
	}{
		{"Tiny (10 words)", "The quick brown fox jumps over the lazy dog again."},
		{"Short (50 words)", strings.Repeat("word ", 50)},
		{"Medium (200 words)", strings.Repeat("word ", 200)},
		{"Long (500 words)", strings.Repeat("word ", 500)},
		{"Very long (1000 words)", strings.Repeat("word ", 1000)},
		{"Code chunk (small)", `
func processData(input []byte) ([]byte, error) {
    parsed, err := parseInput(input)
    if err != nil { return nil, err }
    validated, err := validate(parsed)
    if err != nil { return nil, err }
    return transform(validated), nil
}
`},
		{"Code chunk (medium)", generateCodeChunk(10)},
		{"Code chunk (large)", generateCodeChunk(50)},
		{"Prose (technical)", `Artificial intelligence is transforming the way we live and work. From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly sophisticated. Machine learning algorithms can now process vast amounts of data to identify patterns that would be impossible for humans to detect. Natural language processing has made significant strides, enabling more natural interactions between humans and machines. The future promises even more integration of AI into daily life, with smart homes, personalized education, and advanced robotics becoming commonplace. Deep learning neural networks have revolutionized computer vision, speech recognition, and language understanding. Transformer architectures have enabled breakthroughs in sequence modeling tasks.`},
	}

	fmt.Println("| Test Case | Doc Length | Tokens | Time (μs) | Tokens/sec | Docs/sec |")
	fmt.Println("|-----------|------------|--------|-----------|------------|----------|")

	for _, tc := range testCases {
		// Warmup
		_, _ = tokenizer.Encode(tc.text)

		// Benchmark (more runs for small docs, fewer for large)
		runs := 100
		if len(tc.text) >= 1000 {
			runs = 10
		}

		var times []int64
		var tokenCount int

		for i := 0; i < runs; i++ {
			start := time.Now()
			tokens, _ := tokenizer.Encode(tc.text)
			elapsed := time.Since(start)
			times = append(times, elapsed.Microseconds())
			tokenCount = len(tokens)
		}

		sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
		medianUs := times[len(times)/2]
		medianSecs := float64(medianUs) / 1_000_000.0

		tokensPerSec := float64(tokenCount) / medianSecs
		docsPerSec := 1.0 / medianSecs

		fmt.Printf("| %s | %d | %d | %d | %.0f | %.0f |\n",
			tc.name,
			len(tc.text),
			tokenCount,
			medianUs,
			tokensPerSec,
			docsPerSec,
		)
	}

	fmt.Println()

	// Batch throughput test
	fmt.Println("=== Batch Throughput Test ===")
	fmt.Println()

	batchSizes := []int{10, 100, 1000}
	chunkText := "func example() { let x := 42 }"

	for _, batchSize := range batchSizes {
		chunks := make([]string, batchSize)
		for i := range chunks {
			chunks[i] = chunkText
		}

		start := time.Now()
		totalTokens := 0
		for _, chunk := range chunks {
			tokens, _ := tokenizer.Encode(chunk)
			totalTokens += len(tokens)
		}
		elapsed := time.Since(start)

		docsPerSec := float64(batchSize) / elapsed.Seconds()
		tokensPerSec := float64(totalTokens) / elapsed.Seconds()

		fmt.Printf("Batch size %d: %.0f docs/sec, %.0f tokens/sec (%d total tokens in %v)\n",
			batchSize, docsPerSec, tokensPerSec, totalTokens, elapsed)
	}

	fmt.Println()

	// Algorithmic complexity test - does it scale linearly?
	fmt.Println("=== Scalability Test (Linear vs Quadratic) ===")
	fmt.Println()
	fmt.Println("Testing if tokenization time scales O(n) or O(n²) with document size")
	fmt.Println()

	fmt.Println("| Doc Size (words) | Time (μs) | Tokens | μs per token | Scaling factor |")
	fmt.Println("|------------------|-----------|--------|--------------|----------------|")

	sizes := []int{100, 200, 400, 800, 1600}
	var prevTime int64
	var prevSize int

	for _, size := range sizes {
		text := strings.Repeat("word ", size)

		// Median of 10 runs
		var times []int64
		var tokenCount int
		for i := 0; i < 10; i++ {
			start := time.Now()
			tokens, _ := tokenizer.Encode(text)
			times = append(times, time.Since(start).Microseconds())
			tokenCount = len(tokens)
		}
		sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
		medianUs := times[5]

		usPerToken := float64(medianUs) / float64(tokenCount)

		var scaling string
		if prevTime > 0 {
			sizeRatio := float64(size) / float64(prevSize)
			timeRatio := float64(medianUs) / float64(prevTime)
			scaling = fmt.Sprintf("%.2fx (expected %.2fx for O(n))", timeRatio, sizeRatio)
		} else {
			scaling = "baseline"
		}

		fmt.Printf("| %d | %d | %d | %.2f | %s |\n",
			size, medianUs, tokenCount, usPerToken, scaling)

		prevTime = medianUs
		prevSize = size
	}

	fmt.Println()
	fmt.Println("Note: O(n) scaling means time_ratio ≈ size_ratio")
	fmt.Println("      O(n²) scaling means time_ratio ≈ size_ratio²")
}

func generateCodeChunk(numFunctions int) string {
	var sb strings.Builder
	for i := 0; i < numFunctions; i++ {
		fmt.Fprintf(&sb, `
func processItem%d(data []byte) error {
    parsed, err := parse(data)
    if err != nil { return err }
    if err := validate(parsed); err != nil { return err }
    return transform(parsed)
}
`, i)
	}
	return sb.String()
}
