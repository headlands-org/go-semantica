package kernels

import (
	"runtime"
	"testing"
)

// mockWorkerPool simulates parallel execution
type mockWorkerPool struct {
	workers int
}

func (m *mockWorkerPool) Run(tasks ...func()) {
	// Execute tasks in parallel using goroutines
	if len(tasks) <= 1 {
		for _, task := range tasks {
			task()
		}
		return
	}

	done := make(chan bool, len(tasks))
	for _, task := range tasks {
		task := task // capture
		go func() {
			task()
			done <- true
		}()
	}

	// Wait for all tasks to complete
	for i := 0; i < len(tasks); i++ {
		<-done
	}
}

// BenchmarkAttentionSingleQuery benchmarks single-query attention (typical inference case)
func BenchmarkAttentionSingleQuery(b *testing.B) {
	seqLen := 1
	nHeads := 32
	headDim := 64
	totalDim := nHeads * headDim

	Q := make([]float32, seqLen*totalDim)
	K := make([]float32, seqLen*totalDim)
	V := make([]float32, seqLen*totalDim)

	for i := range Q {
		Q[i] = float32(i%10) * 0.1
		K[i] = float32((i+5)%10) * 0.1
		V[i] = float32((i+2)%10) * 0.1
	}

	pool := &mockWorkerPool{workers: runtime.GOMAXPROCS(0)}
	runner := func(tasks ...func()) {
		pool.Run(tasks...)
	}

	b.Run("Serial", func(b *testing.B) {
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*nHeads)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch)
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		output := make([]float32, seqLen*totalDim)
		scratch := make([]float32, seqLen*seqLen*nHeads)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch, 0, runner, 16)
		}
	})
}

// BenchmarkAttentionMultiQuery benchmarks multi-query attention
func BenchmarkAttentionMultiQuery(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Small_8x8", 8, 8, 64},
		{"Medium_16x16", 16, 16, 64},
		{"Large_32x32", 32, 32, 64},
		{"Long_128x16", 128, 16, 64},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			totalDim := cfg.nHeads * cfg.headDim

			Q := make([]float32, cfg.seqLen*totalDim)
			K := make([]float32, cfg.seqLen*totalDim)
			V := make([]float32, cfg.seqLen*totalDim)

			for i := range Q {
				Q[i] = float32(i%10) * 0.1
				K[i] = float32((i+5)%10) * 0.1
				V[i] = float32((i+2)%10) * 0.1
			}

			pool := &mockWorkerPool{workers: runtime.GOMAXPROCS(0)}
			runner := func(tasks ...func()) {
				pool.Run(tasks...)
			}

			b.Run("Serial", func(b *testing.B) {
				output := make([]float32, cfg.seqLen*totalDim)
				scratch := make([]float32, cfg.seqLen*cfg.seqLen*cfg.nHeads)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MultiHeadAttentionWithScale(output, Q, K, V, 1, cfg.seqLen, cfg.nHeads, cfg.headDim, nil, 1.0, scratch)
				}
			})

			b.Run("Parallel", func(b *testing.B) {
				output := make([]float32, cfg.seqLen*totalDim)
				scratch := make([]float32, cfg.seqLen*cfg.seqLen*cfg.nHeads)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MultiHeadAttentionChunked(output, Q, K, V, 1, cfg.seqLen, cfg.nHeads, cfg.headDim, nil, 1.0, scratch, 0, runner, 8)
				}
			})
		})
	}
}

// BenchmarkAttentionHeadCounts benchmarks different head counts
func BenchmarkAttentionHeadCounts(b *testing.B) {
	headCounts := []int{8, 16, 32}
	seqLen := 1 // Single query (typical inference)
	headDim := 64

	for _, nHeads := range headCounts {
		b.Run(b.Name()+"_"+string(rune('0'+nHeads/8))+"x8Heads", func(b *testing.B) {
			totalDim := nHeads * headDim

			Q := make([]float32, seqLen*totalDim)
			K := make([]float32, seqLen*totalDim)
			V := make([]float32, seqLen*totalDim)

			for i := range Q {
				Q[i] = float32(i%10) * 0.1
				K[i] = float32((i+5)%10) * 0.1
				V[i] = float32((i+2)%10) * 0.1
			}

			pool := &mockWorkerPool{workers: runtime.GOMAXPROCS(0)}
			runner := func(tasks ...func()) {
				pool.Run(tasks...)
			}

			output := make([]float32, seqLen*totalDim)
			scratch := make([]float32, seqLen*seqLen*nHeads)

			b.Run("Serial", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MultiHeadAttentionWithScale(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch)
				}
			})

			b.Run("Parallel", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MultiHeadAttentionChunked(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, 1.0, scratch, 0, runner, 16)
				}
			})
		})
	}
}
