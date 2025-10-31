package kernels

import (
	"fmt"
	"math"
	"testing"
)

// BenchmarkQKNormalization measures Q/K normalization performance
// This simulates the normalization step in Gemma models where Q and K are normalized per-head
func BenchmarkQKNormalization(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		// Single query (typical inference)
		{"SingleQuery_8heads", 1, 8, 64},
		{"SingleQuery_16heads", 1, 16, 64},
		{"SingleQuery_32heads", 1, 32, 64},

		// Short sequences
		{"Short_8x8", 8, 8, 64},
		{"Short_16x16", 16, 16, 64},

		// Medium sequences
		{"Medium_32x16", 32, 16, 64},
		{"Medium_64x16", 64, 16, 64},
		{"Medium_128x16", 128, 16, 64},

		// Large sequences
		{"Large_256x32", 256, 32, 128},
		{"Large_512x32", 512, 32, 128},
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		qk := make([]float32, cfg.seqLen*totalDim)
		weight := make([]float32, cfg.headDim)
		normalized := make([]float32, cfg.seqLen*totalDim)

		// Initialize data
		for i := range qk {
			qk[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(cfg.name, func(b *testing.B) {
			totalWork := cfg.seqLen * cfg.nHeads
			b.ReportMetric(float64(totalWork), "heads")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Normalize each head
				for s := 0; s < cfg.seqLen; s++ {
					for h := 0; h < cfg.nHeads; h++ {
						offset := (s*cfg.nHeads + h) * cfg.headDim
						RMSNorm(
							normalized[offset:offset+cfg.headDim],
							qk[offset:offset+cfg.headDim],
							weight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// BenchmarkQKNormalizationSerial measures serial per-head normalization
func BenchmarkQKNormalizationSerial(b *testing.B) {
	seqLengths := []int{1, 8, 16, 32, 64, 128, 256}
	nHeads := 16
	headDim := 64

	for _, seqLen := range seqLengths {
		name := fmt.Sprintf("seqLen%d", seqLen)
		totalDim := nHeads * headDim
		qk := make([]float32, seqLen*totalDim)
		weight := make([]float32, headDim)
		normalized := make([]float32, seqLen*totalDim)

		for i := range qk {
			qk[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < seqLen; s++ {
					for h := 0; h < nHeads; h++ {
						offset := (s*nHeads + h) * headDim
						RMSNorm(
							normalized[offset:offset+headDim],
							qk[offset:offset+headDim],
							weight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// BenchmarkQKNormalizationHeadCounts measures normalization with different head counts
func BenchmarkQKNormalizationHeadCounts(b *testing.B) {
	headCounts := []int{4, 8, 16, 32, 64}
	seqLen := 32
	headDim := 64

	for _, nHeads := range headCounts {
		name := fmt.Sprintf("heads%d", nHeads)
		totalDim := nHeads * headDim
		qk := make([]float32, seqLen*totalDim)
		weight := make([]float32, headDim)
		normalized := make([]float32, seqLen*totalDim)

		for i := range qk {
			qk[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < seqLen; s++ {
					for h := 0; h < nHeads; h++ {
						offset := (s*nHeads + h) * headDim
						RMSNorm(
							normalized[offset:offset+headDim],
							qk[offset:offset+headDim],
							weight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// BenchmarkPerHeadRMSNorm benchmarks the core per-head normalization
func BenchmarkPerHeadRMSNorm(b *testing.B) {
	headDims := []int{32, 64, 96, 128}

	for _, headDim := range headDims {
		name := fmt.Sprintf("dim%d", headDim)
		src := make([]float32, headDim)
		weight := make([]float32, headDim)
		dst := make([]float32, headDim)

		for i := range src {
			src[i] = float32(i%100) * 0.01
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(headDim * 4 * 3))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RMSNorm(dst, src, weight, 1e-6)
			}
		})
	}
}

// BenchmarkQKNormalizationBatched simulates batched Q/K normalization
func BenchmarkQKNormalizationBatched(b *testing.B) {
	batchSizes := []int{1, 2, 4, 8, 16}
	seqLen := 32
	nHeads := 16
	headDim := 64

	for _, batch := range batchSizes {
		name := fmt.Sprintf("batch%d", batch)
		totalDim := nHeads * headDim
		totalSize := batch * seqLen * totalDim
		qk := make([]float32, totalSize)
		weight := make([]float32, headDim)
		normalized := make([]float32, totalSize)

		for i := range qk {
			qk[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for bIdx := 0; bIdx < batch; bIdx++ {
					batchOffset := bIdx * seqLen * totalDim
					for s := 0; s < seqLen; s++ {
						for h := 0; h < nHeads; h++ {
							offset := batchOffset + (s*nHeads+h)*headDim
							RMSNorm(
								normalized[offset:offset+headDim],
								qk[offset:offset+headDim],
								weight,
								1e-6,
							)
						}
					}
				}
			}
		})
	}
}

// BenchmarkQNormalization specifically benchmarks Q normalization (happens before RoPE)
func BenchmarkQNormalization(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Tiny_1x8x64", 1, 8, 64},
		{"Small_8x8x64", 8, 8, 64},
		{"Medium_32x16x64", 32, 16, 64},
		{"Large_128x16x64", 128, 16, 64},
		{"XLarge_512x32x128", 512, 32, 128},
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		Q := make([]float32, cfg.seqLen*totalDim)
		qNormWeight := make([]float32, cfg.headDim)
		QNormed := make([]float32, cfg.seqLen*totalDim)

		for i := range Q {
			Q[i] = float32(i%100) * 0.01
		}
		for i := range qNormWeight {
			qNormWeight[i] = 1.0
		}

		b.Run(cfg.name, func(b *testing.B) {
			totalOps := cfg.seqLen * cfg.nHeads * cfg.headDim * 2 // Read and write
			b.SetBytes(int64(totalOps * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < cfg.seqLen; s++ {
					for h := 0; h < cfg.nHeads; h++ {
						offset := (s*cfg.nHeads + h) * cfg.headDim
						RMSNorm(
							QNormed[offset:offset+cfg.headDim],
							Q[offset:offset+cfg.headDim],
							qNormWeight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// BenchmarkKNormalization specifically benchmarks K normalization (happens before RoPE)
func BenchmarkKNormalization(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Tiny_1x8x64", 1, 8, 64},
		{"Small_8x8x64", 8, 8, 64},
		{"Medium_32x16x64", 32, 16, 64},
		{"Large_128x16x64", 128, 16, 64},
		{"XLarge_512x32x128", 512, 32, 128},
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		K := make([]float32, cfg.seqLen*totalDim)
		kNormWeight := make([]float32, cfg.headDim)
		KNormed := make([]float32, cfg.seqLen*totalDim)

		for i := range K {
			K[i] = float32(i%100) * 0.01
		}
		for i := range kNormWeight {
			kNormWeight[i] = 1.0
		}

		b.Run(cfg.name, func(b *testing.B) {
			totalOps := cfg.seqLen * cfg.nHeads * cfg.headDim * 2
			b.SetBytes(int64(totalOps * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < cfg.seqLen; s++ {
					for h := 0; h < cfg.nHeads; h++ {
						offset := (s*cfg.nHeads + h) * cfg.headDim
						RMSNorm(
							KNormed[offset:offset+cfg.headDim],
							K[offset:offset+cfg.headDim],
							kNormWeight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// BenchmarkQKNormalizationCombined benchmarks normalizing both Q and K together
func BenchmarkQKNormalizationCombined(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		nHeads  int
		headDim int
	}{
		{"Inference_1x16x64", 1, 16, 64},
		{"ShortSeq_16x16x64", 16, 16, 64},
		{"MediumSeq_64x16x64", 64, 16, 64},
		{"LongSeq_256x32x128", 256, 32, 128},
	}

	for _, cfg := range configs {
		totalDim := cfg.nHeads * cfg.headDim
		Q := make([]float32, cfg.seqLen*totalDim)
		K := make([]float32, cfg.seqLen*totalDim)
		qNormWeight := make([]float32, cfg.headDim)
		kNormWeight := make([]float32, cfg.headDim)
		QNormed := make([]float32, cfg.seqLen*totalDim)
		KNormed := make([]float32, cfg.seqLen*totalDim)

		for i := range Q {
			Q[i] = float32(i%100) * 0.01
			K[i] = float32((i+5)%100) * 0.01
		}
		for i := range qNormWeight {
			qNormWeight[i] = 1.0
			kNormWeight[i] = 1.0
		}

		b.Run(cfg.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Normalize Q
				for s := 0; s < cfg.seqLen; s++ {
					for h := 0; h < cfg.nHeads; h++ {
						offset := (s*cfg.nHeads + h) * cfg.headDim
						RMSNorm(
							QNormed[offset:offset+cfg.headDim],
							Q[offset:offset+cfg.headDim],
							qNormWeight,
							1e-6,
						)
					}
				}
				// Normalize K
				for s := 0; s < cfg.seqLen; s++ {
					for h := 0; h < cfg.nHeads; h++ {
						offset := (s*cfg.nHeads + h) * cfg.headDim
						RMSNorm(
							KNormed[offset:offset+cfg.headDim],
							K[offset:offset+cfg.headDim],
							kNormWeight,
							1e-6,
						)
					}
				}
			}
		})
	}
}

// TestQKNormAccuracy validates that per-head normalization is correct
func TestQKNormAccuracy(t *testing.T) {
	seqLen := 4
	nHeads := 2
	headDim := 64

	totalDim := nHeads * headDim
	qk := make([]float32, seqLen*totalDim)
	weight := make([]float32, headDim)
	normalized := make([]float32, seqLen*totalDim)

	for i := range qk {
		qk[i] = float32(i%10) * 0.1
	}
	for i := range weight {
		weight[i] = 1.0
	}

	// Normalize each head
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			offset := (s*nHeads + h) * headDim
			RMSNorm(
				normalized[offset:offset+headDim],
				qk[offset:offset+headDim],
				weight,
				1e-6,
			)
		}
	}

	// Check that each head is normalized correctly
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			offset := (s*nHeads + h) * headDim

			// Calculate RMS of normalized values
			sumSq := float32(0)
			for i := 0; i < headDim; i++ {
				v := normalized[offset+i]
				sumSq += v * v
			}
			rms := float32(math.Sqrt(float64(sumSq / float32(headDim))))

			// RMS should be approximately 1.0 after normalization
			if math.Abs(float64(rms-1.0)) > 0.1 {
				t.Errorf("Head [%d,%d]: RMS = %f, expected ~1.0", s, h, rms)
			}
		}
	}
}
