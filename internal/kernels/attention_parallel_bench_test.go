package kernels

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
)

// mockWorkerPoolForBench simulates parallel execution for benchmarks
type mockWorkerPoolForBench struct{}

func (m *mockWorkerPoolForBench) Run(tasks ...func()) {
	if len(tasks) <= 1 {
		for _, task := range tasks {
			task()
		}
		return
	}

	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, task := range tasks {
		task := task
		go func() {
			defer wg.Done()
			task()
		}()
	}
	wg.Wait()
}

// BenchmarkAttentionHeadParallelization tests head parallelization strategies
func BenchmarkAttentionHeadParallelization(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		// Single-query scenarios (typical inference)
		{"SingleQuery_8heads", 1, 8, 64},
		{"SingleQuery_16heads", 1, 16, 64},
		{"SingleQuery_32heads", 1, 32, 64},
		{"SingleQuery_64heads", 1, 64, 64},

		// Short sequences
		{"Short_8x8", 8, 8, 64},
		{"Short_8x16", 8, 16, 64},
		{"Short_8x32", 8, 32, 64},

		// Medium sequences
		{"Medium_32x8", 32, 8, 64},
		{"Medium_32x16", 32, 16, 64},
		{"Medium_32x32", 32, 32, 64},
		{"Medium_64x16", 64, 16, 64},

		// Long sequences
		{"Long_128x8", 128, 8, 64},
		{"Long_128x16", 128, 16, 64},
		{"Long_256x16", 256, 16, 64},
		{"Long_512x32", 512, 32, 128},
	}

	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		Q := make([]float32, cfg.seqLen*totalDim)
		K := make([]float32, cfg.seqLen*totalDim)
		V := make([]float32, cfg.seqLen*totalDim)
		output := make([]float32, cfg.seqLen*totalDim)
		scratch := make([]float32, cfg.seqLen*cfg.seqLen*cfg.nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run(cfg.name+"/Serial", func(b *testing.B) {
			totalWork := cfg.seqLen * cfg.seqLen * cfg.nHeads * cfg.headDim
			b.ReportMetric(float64(totalWork), "work")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionWithScale(output, Q, K, V, 1, cfg.seqLen, cfg.nHeads, cfg.headDim, nil, 1.0, scratch)
			}
		})

		b.Run(cfg.name+"/Parallel", func(b *testing.B) {
			totalWork := cfg.seqLen * cfg.seqLen * cfg.nHeads * cfg.headDim
			b.ReportMetric(float64(totalWork), "work")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, cfg.seqLen, cfg.nHeads, cfg.headDim, nil, 1.0, scratch, 0, runner, 8)
			}
		})
	}
}

// BenchmarkAttentionParallelHeadCounts tests scalability with different head counts
func BenchmarkAttentionParallelHeadCounts(b *testing.B) {
	headCounts := []int{4, 8, 16, 32, 64}
	seqLen := 32
	headDim := 64

	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	for _, nHeads := range headCounts {
		name := fmt.Sprintf("heads%d", nHeads)
		totalDim := nHeads * headDim
		Q := make([]float32, seqLen*totalDim)
		K := make([]float32, seqLen*totalDim)
		V := make([]float32, seqLen*totalDim)
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run(name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch)
			}
		})

		b.Run(name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch, 0, runner, 8)
			}
		})
	}
}

// BenchmarkAttentionSeqLengths tests scalability with different sequence lengths
func BenchmarkAttentionSeqLengths(b *testing.B) {
	seqLengths := []int{1, 4, 8, 16, 32, 64, 128, 256}
	nHeads := 16
	headDim := 64

	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	for _, seqLen := range seqLengths {
		name := fmt.Sprintf("seqLen%d", seqLen)
		totalDim := nHeads * headDim
		Q := make([]float32, seqLen*totalDim)
		K := make([]float32, seqLen*totalDim)
		V := make([]float32, seqLen*totalDim)
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run(name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch)
			}
		})

		b.Run(name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch, 0, runner, 8)
			}
		})
	}
}

// BenchmarkAttentionCPUScaling measures attention scaling with different CPU counts
func BenchmarkAttentionCPUScaling(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Small_32x16", 32, 16, 64},
		{"Medium_128x16", 128, 16, 64},
		{"Large_512x32", 512, 32, 128},
	}

	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		Q := make([]float32, cfg.seqLen*totalDim)
		K := make([]float32, cfg.seqLen*totalDim)
		V := make([]float32, cfg.seqLen*totalDim)
		output := make([]float32, cfg.seqLen*totalDim)
		scratch := make([]float32, cfg.seqLen*cfg.seqLen*cfg.nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run(cfg.name, func(b *testing.B) {
			cpuCount := runtime.GOMAXPROCS(0)
			b.ReportMetric(float64(cpuCount), "cpus")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, cfg.seqLen, cfg.nHeads, cfg.headDim, nil, 1.0, scratch, 0, runner, 8)
			}
		})
	}
}

// BenchmarkAttentionThresholds tests different threshold points for parallelization
func BenchmarkAttentionThresholds(b *testing.B) {
	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	// Test around the 16-head threshold for single-query
	singleQueryConfigs := []struct {
		name   string
		nHeads int
	}{
		{"Below_8heads", 8},
		{"Below_12heads", 12},
		{"At_16heads", 16},
		{"Above_24heads", 24},
		{"Above_32heads", 32},
	}

	seqLen := 1
	headDim := 64

	for _, cfg := range singleQueryConfigs {
		totalDim := cfg.nHeads * headDim
		Q := make([]float32, seqLen*totalDim)
		K := make([]float32, seqLen*totalDim)
		V := make([]float32, seqLen*totalDim)
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*cfg.nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run("SingleQuery/"+cfg.name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, cfg.nHeads, headDim, nil, 1.0, scratch)
			}
		})

		b.Run("SingleQuery/"+cfg.name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, cfg.nHeads, headDim, nil, 1.0, scratch, 0, runner, cfg.nHeads)
			}
		})
	}

	// Test around the 8-head threshold for multi-query
	multiQueryConfigs := []struct {
		name   string
		nHeads int
	}{
		{"Below_4heads", 4},
		{"Below_6heads", 6},
		{"At_8heads", 8},
		{"Above_12heads", 12},
		{"Above_16heads", 16},
	}

	seqLen = 32

	for _, cfg := range multiQueryConfigs {
		totalDim := cfg.nHeads * headDim
		Q := make([]float32, seqLen*totalDim)
		K := make([]float32, seqLen*totalDim)
		V := make([]float32, seqLen*totalDim)
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*cfg.nHeads)

		for i := range Q {
			Q[i] = float32(i%10) * 0.1
			K[i] = float32((i+5)%10) * 0.1
			V[i] = float32((i+2)%10) * 0.1
		}

		b.Run("MultiQuery/"+cfg.name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, cfg.nHeads, headDim, nil, 1.0, scratch)
			}
		})

		b.Run("MultiQuery/"+cfg.name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, cfg.nHeads, headDim, nil, 1.0, scratch, 0, runner, 8)
			}
		})
	}
}

// BenchmarkAttentionWithMask tests attention with attention masks
func BenchmarkAttentionWithMask(b *testing.B) {
	seqLen := 64
	nHeads := 16
	headDim := 64

	pool := &mockWorkerPoolForBench{}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	totalDim := nHeads * headDim
	Q := make([]float32, seqLen*totalDim)
	K := make([]float32, seqLen*totalDim)
	V := make([]float32, seqLen*totalDim)
	output := make([]float32, seqLen*totalDim)
	scratch := make([]float32, seqLen*seqLen*nHeads)
	mask := make([]float32, seqLen*seqLen)

	for i := range Q {
		Q[i] = float32(i%10) * 0.1
		K[i] = float32((i+5)%10) * 0.1
		V[i] = float32((i+2)%10) * 0.1
	}

	// Causal mask
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask[i*seqLen+j] = -1e10
			}
		}
	}

	b.Run("WithMask/Serial", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, mask, 1.0, scratch)
		}
	})

	b.Run("WithMask/Parallel", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, mask, 1.0, scratch, 0, runner, 8)
		}
	})

	b.Run("WithoutMask/Serial", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch)
		}
	})

	b.Run("WithoutMask/Parallel", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch, 0, runner, 8)
		}
	})
}
