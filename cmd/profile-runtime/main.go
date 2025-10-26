// profile-runtime is a tool for profiling the embedding runtime with different configurations
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

var (
	modelPath            = flag.String("model", "model/embeddinggemma-300m-Q8_0.gguf", "Path to GGUF model")
	batchSize            = flag.Int("batch", 32, "Batch size for processing")
	numWorkers           = flag.Int("workers", 0, "Number of workers (0 = NumCPU)")
	disableMatmulParallel = flag.Bool("disable-matmul-parallel", false, "Disable internal matmul parallelism")
	cpuProfile           = flag.String("cpuprofile", "", "Write CPU profile to file")
	memProfile           = flag.String("memprofile", "", "Write memory profile to file")
	iterations           = flag.Int("iterations", 100, "Number of iterations to run")
	verbose              = flag.Bool("verbose", false, "Enable verbose logging")
)

func main() {
	flag.Parse()

	// Validate model exists
	if _, err := os.Stat(*modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s", *modelPath)
	}

	// Set worker count
	workers := *numWorkers
	if workers == 0 {
		workers = runtime.NumCPU()
	}

	// Print configuration
	fmt.Printf("=== Profiling Configuration ===\n")
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Batch Size: %d\n", *batchSize)
	fmt.Printf("Workers: %d (NumCPU: %d)\n", workers, runtime.NumCPU())
	fmt.Printf("DisableMatmulParallel: %v\n", *disableMatmulParallel)
	fmt.Printf("Iterations: %d\n", *iterations)
	fmt.Printf("CPU Profile: %s\n", *cpuProfile)
	fmt.Printf("Memory Profile: %s\n", *memProfile)
	fmt.Printf("================================\n\n")

	// Open model with specified configuration
	rt, err := ggufembed.Open(*modelPath,
		ggufembed.WithThreads(workers),
		ggufembed.WithDisableMatmulParallel(*disableMatmulParallel),
		ggufembed.WithVerbose(*verbose),
	)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	fmt.Printf("Model loaded: embed_dim=%d, max_seq_len=%d\n\n", rt.EmbedDim(), rt.MaxSeqLen())

	// Prepare test data - variety of text lengths
	texts := generateTestTexts(*batchSize)

	// Warmup run
	fmt.Printf("Running warmup...\n")
	ctx := context.Background()
	_, err = rt.Embed(ctx, texts)
	if err != nil {
		log.Fatalf("Warmup failed: %v", err)
	}
	fmt.Printf("Warmup complete\n\n")

	// Start CPU profiling if requested
	if *cpuProfile != "" {
		// Create directory if needed
		dir := filepath.Dir(*cpuProfile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create profile directory: %v", err)
		}

		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatalf("Failed to create CPU profile: %v", err)
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("Failed to start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
		fmt.Printf("CPU profiling started: %s\n", *cpuProfile)
	}

	// Run benchmark
	fmt.Printf("Running %d iterations...\n", *iterations)
	start := time.Now()

	for i := 0; i < *iterations; i++ {
		_, err := rt.Embed(ctx, texts)
		if err != nil {
			log.Fatalf("Iteration %d failed: %v", i, err)
		}

		if (i+1)%10 == 0 {
			fmt.Printf("  Completed %d/%d iterations\n", i+1, *iterations)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\nBenchmark complete!\n")
	fmt.Printf("Total time: %v\n", elapsed)
	fmt.Printf("Avg time per iteration: %v\n", elapsed/time.Duration(*iterations))
	fmt.Printf("Avg time per text: %v\n", elapsed/time.Duration(*iterations * *batchSize))

	// Write memory profile if requested
	if *memProfile != "" {
		// Create directory if needed
		dir := filepath.Dir(*memProfile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create profile directory: %v", err)
		}

		f, err := os.Create(*memProfile)
		if err != nil {
			log.Fatalf("Failed to create memory profile: %v", err)
		}
		defer f.Close()

		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("Failed to write memory profile: %v", err)
		}
		fmt.Printf("\nMemory profile written: %s\n", *memProfile)
	}

	fmt.Printf("\nProfile collection complete!\n")
}

// generateTestTexts creates a batch of test texts with varying lengths
func generateTestTexts(count int) []string {
	texts := make([]string, count)

	// Mix of short, medium, and long texts
	templates := []string{
		// Short (1-5 tokens)
		"Hello world",
		"Test",
		"Machine learning",

		// Medium (10-20 tokens)
		"This is a test sentence for benchmarking the embedding model performance.",
		"Natural language processing enables computers to understand human language.",
		"Deep learning models can learn complex patterns from large datasets.",

		// Long (30-50 tokens)
		"The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet and is commonly used for testing text processing systems and fonts.",
		"Artificial intelligence and machine learning are transforming how we interact with technology. From voice assistants to recommendation systems, these technologies are becoming increasingly prevalent in our daily lives.",
		"Climate change represents one of the most significant challenges facing humanity today. Rising global temperatures, extreme weather events, and changing precipitation patterns are affecting ecosystems and human societies worldwide.",
	}

	for i := 0; i < count; i++ {
		texts[i] = templates[i%len(templates)]
	}

	return texts
}
