package kernels

import (
	"math"
	"sync"
	"testing"
)

// Simple worker pool for testing
func runTasksSerial(tasks ...func()) {
	for _, task := range tasks {
		if task != nil {
			task()
		}
	}
}

func runTasksParallel(tasks ...func()) {
	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, task := range tasks {
		task := task
		go func() {
			defer wg.Done()
			if task != nil {
				task()
			}
		}()
	}
	wg.Wait()
}

func TestApplyRoPECachedParallel(t *testing.T) {
	tests := []struct {
		name     string
		seqLen   int
		nHeads   int
		headDim  int
		runTasks func(...func())
	}{
		{
			name:     "Small_4x8_Serial",
			seqLen:   4,
			nHeads:   8,
			headDim:  64,
			runTasks: runTasksSerial,
		},
		{
			name:     "Threshold_8x8_Serial",
			seqLen:   8,
			nHeads:   8,
			headDim:  64,
			runTasks: runTasksSerial,
		},
		{
			name:     "Medium_16x16_Parallel",
			seqLen:   16,
			nHeads:   16,
			headDim:  64,
			runTasks: runTasksParallel,
		},
		{
			name:     "Large_64x32_Parallel",
			seqLen:   64,
			nHeads:   32,
			headDim:  128,
			runTasks: runTasksParallel,
		},
		{
			name:     "FewHeads_32x4_Parallel",
			seqLen:   32,
			nHeads:   4,
			headDim:  64,
			runTasks: runTasksParallel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create RoPE cache
			cache := NewRoPECache(tt.headDim, 10000.0, 2048)

			// Create position array
			pos := make([]int, tt.seqLen)
			for i := 0; i < tt.seqLen; i++ {
				pos[i] = i
			}

			// Create two identical input tensors
			size := tt.seqLen * tt.nHeads * tt.headDim
			qkSerial := make([]float32, size)
			qkParallel := make([]float32, size)

			// Fill with test data (use varying values to detect errors)
			for i := 0; i < size; i++ {
				val := float32(math.Sin(float64(i) * 0.1))
				qkSerial[i] = val
				qkParallel[i] = val
			}

			// Apply serial version
			ApplyRoPECached(qkSerial, tt.seqLen, tt.nHeads, tt.headDim, pos, cache)

			// Apply parallel version with default threshold (64)
			ApplyRoPECachedParallel(qkParallel, tt.seqLen, tt.nHeads, tt.headDim, pos, cache, tt.runTasks, 64)

			// Compare results
			maxDiff := float32(0.0)
			for i := 0; i < size; i++ {
				diff := float32(math.Abs(float64(qkSerial[i] - qkParallel[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Results should be identical (within floating point precision)
			if maxDiff > 1e-6 {
				t.Errorf("Results differ: max diff = %e", maxDiff)
			} else {
				t.Logf("Results match (max diff: %e)", maxDiff)
			}
		})
	}
}

func TestApplyRoPECachedParallelThreshold(t *testing.T) {
	cache := NewRoPECache(64, 10000.0, 2048)

	tests := []struct {
		name          string
		seqLen        int
		nHeads        int
		shouldSerialize bool
	}{
		{"BelowThreshold_4x4", 4, 4, true},    // 16 < 64
		{"BelowThreshold_8x4", 8, 4, true},    // 32 < 64
		{"AtThreshold_8x8", 8, 8, false},      // 64 >= 64
		{"AboveThreshold_16x8", 16, 8, false}, // 128 >= 64
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pos := make([]int, tt.seqLen)
			for i := 0; i < tt.seqLen; i++ {
				pos[i] = i
			}

			size := tt.seqLen * tt.nHeads * 64
			qk := make([]float32, size)
			for i := 0; i < size; i++ {
				qk[i] = float32(i)
			}

			// Track if runTasks is called
			runTasksCalled := false
			runTasksWrapper := func(tasks ...func()) {
				if len(tasks) > 0 {
					runTasksCalled = true
				}
				runTasksSerial(tasks...)
			}

			ApplyRoPECachedParallel(qk, tt.seqLen, tt.nHeads, 64, pos, cache, runTasksWrapper, 64)

			// Verify threshold behavior
			totalWork := tt.seqLen * tt.nHeads
			if totalWork < 64 && runTasksCalled {
				t.Errorf("runTasks called for small workload (total=%d)", totalWork)
			}
			if totalWork >= 64 && !runTasksCalled {
				t.Errorf("runTasks not called for large workload (total=%d)", totalWork)
			}
		})
	}
}

func BenchmarkApplyRoPECached(b *testing.B) {
	benchmarks := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Short_4x8x64", 4, 8, 64},
		{"Short_8x8x64", 8, 8, 64},
		{"Medium_32x16x64", 32, 16, 64},
		{"Long_128x16x64", 128, 16, 64},
		{"Long_256x32x128", 256, 32, 128},
	}

	for _, bm := range benchmarks {
		cache := NewRoPECache(bm.headDim, 10000.0, 2048)
		size := bm.seqLen * bm.nHeads * bm.headDim
		qk := make([]float32, size)
		pos := make([]int, bm.seqLen)
		for i := 0; i < bm.seqLen; i++ {
			pos[i] = i
		}

		b.Run(bm.name+"_Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECached(qk, bm.seqLen, bm.nHeads, bm.headDim, pos, cache)
			}
		})

		b.Run(bm.name+"_Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECachedParallel(qk, bm.seqLen, bm.nHeads, bm.headDim, pos, cache, runTasksParallel, 64)
			}
		})
	}
}
