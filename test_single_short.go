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

func measureLatency(rt ggufembed.Runtime, doc string, name string) {
	ctx := context.Background()

	// Warmup
	fmt.Printf("\n%s - Warmup (5 runs)...\n", name)
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingle(ctx, doc)
	}

	// Measure 20 runs
	fmt.Printf("%s - Measuring (20 runs)...\n", name)
	var latencies []float64
	for i := 0; i < 20; i++ {
		start := time.Now()
		_, err := rt.EmbedSingle(ctx, doc)
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

	fmt.Printf("\n%s Results:\n", name)
	fmt.Printf("  P50: %.1f ms\n", p50)
	fmt.Printf("  P95: %.1f ms\n", p95)
	fmt.Printf("  P99: %.1f ms\n", p99)
}

func main() {
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"
	shortDoc := "The quick brown fox jumps over the lazy dog"

	// Test with mmap (default)
	fmt.Println("=== Testing with MMAP (zero-copy) ===")
	rtMmap, err := ggufembed.Open(modelPath, ggufembed.WithMmap(true))
	if err != nil {
		log.Fatal(err)
	}
	measureLatency(rtMmap, shortDoc, "MMAP")
	rtMmap.Close()

	// Test with RAM loading
	fmt.Println("\n\n=== Testing with RAM (loaded into memory) ===")
	rtRAM, err := ggufembed.Open(modelPath, ggufembed.WithMmap(false))
	if err != nil {
		log.Fatal(err)
	}
	measureLatency(rtRAM, shortDoc, "RAM")
	rtRAM.Close()

	os.Exit(0)
}
