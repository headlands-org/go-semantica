package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/lth/pure-go-llamas/internal/gguf"
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	// Get file size
	info, err := os.Stat(modelPath)
	if err != nil {
		log.Fatalf("stat file: %v", err)
	}
	fileSize := float64(info.Size()) / (1024 * 1024)
	fmt.Printf("File size: %.2f MB\n\n", fileSize)

	// Benchmark GGUF reader open (just mmap + parse)
	fmt.Println("=== Phase 1: GGUF Reader Open (mmap + metadata parse) ===")
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	start := time.Now()
	reader, err := gguf.Open(modelPath)
	if err != nil {
		log.Fatalf("open gguf: %v", err)
	}
	elapsed := time.Since(start)

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	memDelta := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	fmt.Printf("Time: %v\n", elapsed)
	fmt.Printf("Memory allocated: %.2f MB\n", memDelta)
	fmt.Printf("Header: %d tensors, %d metadata keys\n\n", reader.Header().TensorCount, reader.Header().MetadataKVSize)

	reader.Close()

	// Benchmark full model load
	fmt.Println("=== Phase 2: Full Model Load (tokenizer + weight indexing) ===")
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	start = time.Now()
	model, err := ggufembed.Open(modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	elapsed = time.Since(start)

	runtime.ReadMemStats(&memAfter)
	memDelta = float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	fmt.Printf("Time: %v\n", elapsed)
	fmt.Printf("Memory allocated: %.2f MB\n\n", memDelta)

	// Benchmark first inference (cold cache)
	fmt.Println("=== Phase 3: First Inference (cold cache) ===")
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	start = time.Now()
	ctx := context.Background()
	_, err = model.EmbedSingle(ctx, "Hello world")
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	elapsed = time.Since(start)

	runtime.ReadMemStats(&memAfter)
	memDelta = float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	fmt.Printf("Time: %v\n", elapsed)
	fmt.Printf("Memory allocated: %.2f MB\n\n", memDelta)

	// Benchmark second inference (warm cache)
	fmt.Println("=== Phase 4: Second Inference (warm cache) ===")
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	start = time.Now()
	_, err = model.EmbedSingle(ctx, "Hello world")
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	elapsed = time.Since(start)

	runtime.ReadMemStats(&memAfter)
	memDelta = float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	fmt.Printf("Time: %v\n", elapsed)
	fmt.Printf("Memory allocated: %.2f MB\n\n", memDelta)

	model.Close()

	fmt.Println("=== Analysis ===")
	fmt.Println("If Phase 1 allocates ~file size MB, the mmap is being copied to RAM")
	fmt.Println("Expected: Phase 1 should allocate <10 MB (just metadata)")
	fmt.Println("Actual file should stay in OS page cache, not Go heap")
}
