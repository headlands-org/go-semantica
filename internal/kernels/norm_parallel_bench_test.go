package kernels

import (
	"fmt"
	"testing"
)

// BenchmarkRMSNormSizes measures RMSNorm performance across different vector sizes
func BenchmarkRMSNormSizes(b *testing.B) {
	sizes := []int{
		128,   // Small embedding
		256,   // Small
		512,   // Medium
		1024,  // Medium-large
		2048,  // Large (common for Gemma)
		4096,  // Very large
		8192,  // Extra large
	}

	for _, n := range sizes {
		name := fmt.Sprintf("dim%d", n)
		src := make([]float32, n)
		weight := make([]float32, n)
		dst := make([]float32, n)

		for i := range src {
			src[i] = float32(i%100) * 0.01
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(n * 4 * 3)) // 3 arrays of float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RMSNorm(dst, src, weight, 1e-6)
			}
		})
	}
}

// BenchmarkRMSNormBatchSizes measures RMSNorm performance for batched operations
func BenchmarkRMSNormBatchSizes(b *testing.B) {
	configs := []struct {
		batchSize int
		dim       int
	}{
		{1, 2048},   // Single item, typical dim
		{4, 2048},   // Small batch
		{8, 2048},   // Medium batch
		{16, 2048},  // Large batch
		{32, 2048},  // Very large batch
		{1, 4096},   // Single item, large dim
		{4, 4096},   // Small batch, large dim
		{16, 4096},  // Large batch, large dim
	}

	for _, cfg := range configs {
		name := fmt.Sprintf("batch%d_dim%d", cfg.batchSize, cfg.dim)
		totalSize := cfg.batchSize * cfg.dim
		src := make([]float32, totalSize)
		weight := make([]float32, cfg.dim)
		dst := make([]float32, totalSize)

		for i := range src {
			src[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(totalSize * 4 * 2)) // src + dst
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Process each batch item
				for j := 0; j < cfg.batchSize; j++ {
					offset := j * cfg.dim
					RMSNorm(dst[offset:offset+cfg.dim], src[offset:offset+cfg.dim], weight, 1e-6)
				}
			}
		})
	}
}

// BenchmarkRMSNormVsSIMD compares scalar vs SIMD implementations
func BenchmarkRMSNormVsSIMD(b *testing.B) {
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		name := fmt.Sprintf("dim%d", n)
		src := make([]float32, n)
		weight := make([]float32, n)
		dst := make([]float32, n)

		for i := range src {
			src[i] = float32(i%100) * 0.01
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RMSNorm(dst, src, weight, 1e-6)
			}
		})
	}
}

// BenchmarkLayerNormSizes measures LayerNorm performance across different sizes
func BenchmarkLayerNormSizes(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		name := fmt.Sprintf("dim%d", n)
		src := make([]float32, n)
		gamma := make([]float32, n)
		beta := make([]float32, n)
		dst := make([]float32, n)

		for i := range src {
			src[i] = float32(i%100) * 0.01
			gamma[i] = 1.0
			beta[i] = 0.0
		}

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(n * 4 * 4)) // 4 arrays
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				LayerNorm(dst, src, gamma, beta, 1e-6)
			}
		})
	}
}

// BenchmarkRMSNormGemmaSizes measures Gemma-specific RMSNorm
func BenchmarkRMSNormGemmaSizes(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		name := fmt.Sprintf("dim%d", n)
		src := make([]float32, n)
		weight := make([]float32, n)
		dst := make([]float32, n)

		for i := range src {
			src[i] = float32(i%100) * 0.01
			weight[i] = 0.1 // Gemma typically uses small weight values
		}

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RMSNormGemma(dst, src, weight, 1e-6)
			}
		})
	}
}

// BenchmarkNormalizationComparison compares different normalization methods
func BenchmarkNormalizationComparison(b *testing.B) {
	n := 2048 // Typical embedding dimension
	src := make([]float32, n)
	weight := make([]float32, n)
	gamma := make([]float32, n)
	beta := make([]float32, n)
	dst := make([]float32, n)

	for i := range src {
		src[i] = float32(i%100) * 0.01
		weight[i] = 1.0
		gamma[i] = 1.0
		beta[i] = 0.0
	}

	b.Run("RMSNorm", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			RMSNorm(dst, src, weight, 1e-6)
		}
	})

	b.Run("LayerNorm", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			LayerNorm(dst, src, gamma, beta, 1e-6)
		}
	})

	b.Run("RMSNormGemma", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			RMSNormGemma(dst, src, weight, 1e-6)
		}
	})
}

// BenchmarkRMSNormSequentialBatch simulates sequential processing of batch items
func BenchmarkRMSNormSequentialBatch(b *testing.B) {
	batchSizes := []int{1, 2, 4, 8, 16, 32}
	dim := 2048

	for _, batchSize := range batchSizes {
		name := fmt.Sprintf("batch%d", batchSize)
		totalSize := batchSize * dim
		src := make([]float32, totalSize)
		weight := make([]float32, dim)
		dst := make([]float32, totalSize)

		for i := range src {
			src[i] = float32(i%100) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := 0; j < batchSize; j++ {
					offset := j * dim
					RMSNorm(dst[offset:offset+dim], src[offset:offset+dim], weight, 1e-6)
				}
			}
		})
	}
}
