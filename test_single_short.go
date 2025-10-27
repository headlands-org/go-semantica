package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"
	shortDoc := "The quick brown fox jumps over the lazy dog"

	rt, err := ggufembed.Open(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()

	ctx := context.Background()

	// Warmup
	fmt.Println("Warmup (5 runs)...")
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingle(ctx, shortDoc)
	}

	// Measure 20 runs
	fmt.Println("Measuring (20 runs)...")
	var latencies []float64
	for i := 0; i < 20; i++ {
		start := time.Now()
		_, err := rt.EmbedSingle(ctx, shortDoc)
		if err != nil {
			log.Fatal(err)
		}
		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		latencies = append(latencies, latencyMs)
	}

	// Sort for percentiles
	sort.Float64s(latencies)

	// Calculate percentiles
	p50 := latencies[len(latencies)/2]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]

	fmt.Printf("\nResults (model already loaded):\n")
	fmt.Printf("  P50: %.1f ms\n", p50)
	fmt.Printf("  P95: %.1f ms\n", p95)
	fmt.Printf("  P99: %.1f ms\n", p99)
}
