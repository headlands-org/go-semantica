package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func runBenchmark(modelPath string, useMmap bool) {
	mode := "MMAP"
	if !useMmap {
		mode = "RAM"
	}

	fmt.Printf("\n=== %s MODE ===\n\n", mode)

	rt, err := ggufembed.Open(modelPath, ggufembed.WithMmap(useMmap))
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()

	ctx := context.Background()

	shortDoc := "The quick brown fox jumps over the lazy dog"
	longDoc := "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."

	// Test 1: Short document
	fmt.Println("Short document (9 words):")
	fmt.Println("  Warmup (5 runs)...")
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingle(ctx, shortDoc)
	}

	fmt.Println("  Measuring (20 runs)...")
	var shortLatencies []float64
	for i := 0; i < 20; i++ {
		start := time.Now()
		_, err := rt.EmbedSingle(ctx, shortDoc)
		if err != nil {
			log.Fatal(err)
		}
		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		shortLatencies = append(shortLatencies, latencyMs)
	}
	sort.Float64s(shortLatencies)
	fmt.Printf("  P50: %.1f ms\n", shortLatencies[len(shortLatencies)/2])

	// Test 2: Long document
	fmt.Println("\nLong document (49 words):")
	fmt.Println("  Warmup (5 runs)...")
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingle(ctx, longDoc)
	}

	fmt.Println("  Measuring (20 runs)...")
	var longLatencies []float64
	for i := 0; i < 20; i++ {
		start := time.Now()
		_, err := rt.EmbedSingle(ctx, longDoc)
		if err != nil {
			log.Fatal(err)
		}
		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		longLatencies = append(longLatencies, latencyMs)
	}
	sort.Float64s(longLatencies)
	fmt.Printf("  P50: %.1f ms\n", longLatencies[len(longLatencies)/2])
}

func main() {
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"

	runBenchmark(modelPath, true)  // MMAP
	runBenchmark(modelPath, false) // RAM

	os.Exit(0)
}
