package runtime

import (
	"runtime"
	"testing"
)

func TestMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	// Force GC before loading
	runtime.GC()

	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	t.Logf("Before LoadModel:")
	t.Logf("  Alloc: %.2f MB", float64(m1.Alloc)/1024/1024)
	t.Logf("  HeapAlloc: %.2f MB", float64(m1.HeapAlloc)/1024/1024)

	// Load model
	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	t.Logf("\nAfter LoadModel:")
	t.Logf("  Alloc: %.2f MB", float64(m2.Alloc)/1024/1024)
	t.Logf("  HeapAlloc: %.2f MB", float64(m2.HeapAlloc)/1024/1024)

	delta := float64(m2.HeapAlloc-m1.HeapAlloc) / 1024 / 1024
	t.Logf("\nMemory increase from model: %.2f MB", delta)
	t.Logf("GGUF file size: ~314 MB")
	t.Logf("Expected: GGUF (~314 MB) + tokenizer (~90 MB) + norms (~10 MB) + scales (~10 MB) = ~424 MB")

	// Zero-copy target: GGUF size + tokenizer + norms + pre-converted scales
	// Scales: 18 layers × 7 weights × ~10KB each ≈ 10 MB
	if delta > 460 {
		t.Errorf("❌ Memory usage too high: %.2f MB > 460 MB", delta)
	} else {
		t.Logf("✅ Zero-copy SUCCESS! Memory: %.2f MB (GGUF + tokenizer + norms + scales)", delta)
	}

	// Run inference
	tokenIDs := []int{2, 9259, 1902, 1}
	_, err = model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runtime.GC()
	var m3 runtime.MemStats
	runtime.ReadMemStats(&m3)
	t.Logf("\nAfter Forward():")
	t.Logf("  Alloc: %.2f MB", float64(m3.Alloc)/1024/1024)
	t.Logf("  HeapAlloc: %.2f MB", float64(m3.HeapAlloc)/1024/1024)

	totalDelta := float64(m3.HeapAlloc-m1.HeapAlloc) / 1024 / 1024
	t.Logf("\nTotal memory footprint: %.2f MB", totalDelta)

	// Print breakdown
	t.Logf("\nZero-copy architecture:")
	t.Logf("  GGUF file in memory: ~314 MB (loaded once)")
	t.Logf("  Token embeddings: 0 MB (slice view into GGUF)")
	t.Logf("  Layer weights: 0 MB (slice views into GGUF)")
	t.Logf("  Tokenizer vocab: ~90 MB")
	t.Logf("  FP32 norm weights: ~5-10 MB")
	t.Logf("  Pre-converted scales: ~10 MB (18 layers × 7 weights)")
	t.Logf("  Total expected: ~424 MB")
}

func BenchmarkMemoryForward(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	tokenIDs := []int{2, 9259, 1902, 1}

	// Warm up
	model.Forward(tokenIDs)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = model.Forward(tokenIDs)
	}

	// Report memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	b.ReportMetric(float64(m.Alloc)/1024/1024, "MB/alloc")
}
