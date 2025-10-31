package kernels

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
)

// BenchmarkRoPEParallelVsSerial provides comprehensive comparison of serial vs parallel RoPE
// with various workload sizes to identify optimal thresholds
func BenchmarkRoPEParallelVsSerial(b *testing.B) {
	configs := []struct {
		seqLen  int
		nHeads  int
		headDim int
	}{
		// Below threshold (totalWork < 64)
		{4, 8, 64},  // totalWork = 32
		{8, 4, 64},  // totalWork = 32
		{4, 12, 64}, // totalWork = 48

		// At threshold (totalWork = 64)
		{8, 8, 64},  // totalWork = 64
		{16, 4, 64}, // totalWork = 64

		// Above threshold - small
		{16, 8, 64}, // totalWork = 128
		{32, 4, 64}, // totalWork = 128
		{8, 16, 64}, // totalWork = 128

		// Medium workloads
		{32, 16, 64}, // totalWork = 512
		{64, 8, 64},  // totalWork = 512
		{128, 8, 64}, // totalWork = 1024

		// Large workloads
		{256, 16, 64},  // totalWork = 4096
		{512, 16, 64},  // totalWork = 8192
		{256, 32, 128}, // totalWork = 8192, larger headDim
	}

	runTasks := func(tasks ...func()) {
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

	for _, cfg := range configs {
		name := fmt.Sprintf("seqLen%d_heads%d_dim%d", cfg.seqLen, cfg.nHeads, cfg.headDim)
		totalWork := cfg.seqLen * cfg.nHeads

		cache := NewRoPECache(cfg.headDim, 10000.0, 2048)
		size := cfg.seqLen * cfg.nHeads * cfg.headDim
		qk := make([]float32, size)
		pos := make([]int, cfg.seqLen)
		for i := 0; i < cfg.seqLen; i++ {
			pos[i] = i
			for j := i * cfg.nHeads * cfg.headDim; j < (i+1)*cfg.nHeads*cfg.headDim; j++ {
				qk[j] = float32(j%100) * 0.01
			}
		}

		b.Run(name+"/Serial", func(b *testing.B) {
			b.ReportMetric(float64(totalWork), "work")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECached(qk, cfg.seqLen, cfg.nHeads, cfg.headDim, pos, cache)
			}
		})

		b.Run(name+"/Parallel", func(b *testing.B) {
			b.ReportMetric(float64(totalWork), "work")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECachedParallel(qk, cfg.seqLen, cfg.nHeads, cfg.headDim, pos, cache, runTasks, 64)
			}
		})
	}
}

// BenchmarkRoPEScaling measures RoPE scaling across different CPU counts
func BenchmarkRoPEScaling(b *testing.B) {
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

	runTasks := func(tasks ...func()) {
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

	for _, cfg := range configs {
		cache := NewRoPECache(cfg.headDim, 10000.0, 2048)
		size := cfg.seqLen * cfg.nHeads * cfg.headDim
		qk := make([]float32, size)
		pos := make([]int, cfg.seqLen)
		for i := 0; i < cfg.seqLen; i++ {
			pos[i] = i
		}

		b.Run(cfg.name, func(b *testing.B) {
			cpuCount := runtime.GOMAXPROCS(0)
			b.ReportMetric(float64(cpuCount), "cpus")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECachedParallel(qk, cfg.seqLen, cfg.nHeads, cfg.headDim, pos, cache, runTasks, 64)
			}
		})
	}
}

// BenchmarkRoPEHeadVariations tests performance with different head counts
func BenchmarkRoPEHeadVariations(b *testing.B) {
	headCounts := []int{4, 8, 16, 32, 64}
	seqLen := 128
	headDim := 64

	runTasks := func(tasks ...func()) {
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

	for _, nHeads := range headCounts {
		name := fmt.Sprintf("heads%d", nHeads)
		cache := NewRoPECache(headDim, 10000.0, 2048)
		size := seqLen * nHeads * headDim
		qk := make([]float32, size)
		pos := make([]int, seqLen)
		for i := 0; i < seqLen; i++ {
			pos[i] = i
		}

		b.Run(name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECached(qk, seqLen, nHeads, headDim, pos, cache)
			}
		})

		b.Run(name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECachedParallel(qk, seqLen, nHeads, headDim, pos, cache, runTasks, 64)
			}
		})
	}
}

// BenchmarkRoPESeqLenVariations tests performance with different sequence lengths
func BenchmarkRoPESeqLenVariations(b *testing.B) {
	seqLengths := []int{1, 4, 8, 16, 32, 64, 128, 256, 512}
	nHeads := 16
	headDim := 64

	runTasks := func(tasks ...func()) {
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

	for _, seqLen := range seqLengths {
		name := fmt.Sprintf("seqLen%d", seqLen)
		cache := NewRoPECache(headDim, 10000.0, 2048)
		size := seqLen * nHeads * headDim
		qk := make([]float32, size)
		pos := make([]int, seqLen)
		for i := 0; i < seqLen; i++ {
			pos[i] = i
		}

		b.Run(name+"/Serial", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECached(qk, seqLen, nHeads, headDim, pos, cache)
			}
		})

		b.Run(name+"/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ApplyRoPECachedParallel(qk, seqLen, nHeads, headDim, pos, cache, runTasks, 64)
			}
		})
	}
}
