package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime/pprof"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
	// Start CPU profiling
	f, err := os.Create("single_long_cpu.prof")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal(err)
	}
	defer pprof.StopCPUProfile()

	// Load model
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"
	rt, err := ggufembed.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	// Long document (49 words)
	longDoc := "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."

	ctx := context.Background()

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 3; i++ {
		_, _ = rt.EmbedSingle(ctx, longDoc)
	}

	// Run 50 iterations to get good profiling data
	fmt.Println("Running 50 single long document embeddings...")
	for i := 0; i < 50; i++ {
		_, err := rt.EmbedSingle(ctx, longDoc)
		if err != nil {
			log.Printf("Error on iteration %d: %v", i, err)
		}
		if (i+1)%10 == 0 {
			fmt.Printf("  Completed %d/50\n", i+1)
		}
	}

	fmt.Println("\nDone! Profile saved to single_long_cpu.prof")
	fmt.Println("\nAnalyze with:")
	fmt.Println("  go tool pprof -top single_long_cpu.prof")
	fmt.Println("  go tool pprof -http=:8080 single_long_cpu.prof")
}
