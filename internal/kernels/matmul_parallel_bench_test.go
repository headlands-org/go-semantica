package kernels

import (
	"fmt"
	"testing"
)

// BenchmarkMatMulGGMLTiling measures the effect of cache-friendly blocking
func BenchmarkMatMulGGMLTiling(b *testing.B) {
	configs := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		// Small matrices
		{"Small_1x128x128", 1, 128, 128},
		{"Small_1x256x256", 1, 256, 256},

		// Medium matrices (typical for layers)
		{"Medium_1x512x512", 1, 512, 512},
		{"Medium_1x1024x1024", 1, 1024, 1024},
		{"Medium_4x512x512", 4, 512, 512},

		// Large matrices
		{"Large_1x2048x2048", 1, 2048, 2048},
		{"Large_4x2048x2048", 4, 2048, 2048},
		{"Large_8x2048x2048", 8, 2048, 2048},

		// Asymmetric matrices (common in transformers)
		{"Asymmetric_1x512x2048", 1, 512, 2048},
		{"Asymmetric_1x2048x512", 1, 2048, 512},
		{"Asymmetric_4x512x2048", 4, 512, 2048},
		{"Asymmetric_4x2048x8192", 4, 2048, 8192},

		// Batch variations
		{"Batch1_512x2048", 1, 512, 2048},
		{"Batch2_512x2048", 2, 512, 2048},
		{"Batch4_512x2048", 4, 512, 2048},
		{"Batch8_512x2048", 8, 512, 2048},
		{"Batch16_512x2048", 16, 512, 2048},
		{"Batch32_512x2048", 32, 512, 2048},
	}

	for _, cfg := range configs {
		weight := make([]float32, cfg.outDim*cfg.inDim)
		input := make([]float32, cfg.batch*cfg.inDim)
		dst := make([]float32, cfg.batch*cfg.outDim)

		// Initialize with non-zero values
		for i := range weight {
			weight[i] = float32(i%100) * 0.01
		}
		for i := range input {
			input[i] = float32(i%100) * 0.01
		}

		b.Run(cfg.name, func(b *testing.B) {
			ops := int64(2 * cfg.batch * cfg.inDim * cfg.outDim)
			b.SetBytes(ops * 4) // Approximate data movement
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulGGML(dst, weight, input, cfg.batch, cfg.inDim, cfg.outDim)
			}
			b.ReportMetric(float64(ops)/1e9, "gops")
		})
	}
}

// BenchmarkMatMulF32Tiling measures standard matmul with different block sizes
func BenchmarkMatMulF32Tiling(b *testing.B) {
	configs := []struct {
		name string
		M, K, N int
	}{
		{"Square_128", 128, 128, 128},
		{"Square_256", 256, 256, 256},
		{"Square_512", 512, 512, 512},
		{"Square_1024", 1024, 1024, 1024},

		{"Tall_256x1024x256", 256, 1024, 256},
		{"Tall_512x2048x512", 512, 2048, 512},

		{"Wide_256x256x1024", 256, 256, 1024},
		{"Wide_512x512x2048", 512, 512, 2048},
	}

	for _, cfg := range configs {
		a := make([]float32, cfg.M*cfg.K)
		mat := make([]float32, cfg.K*cfg.N)
		dst := make([]float32, cfg.M*cfg.N)

		for i := range a {
			a[i] = float32(i%100) * 0.01
		}
		for i := range mat {
			mat[i] = float32(i%100) * 0.01
		}

		b.Run(cfg.name, func(b *testing.B) {
			ops := int64(2 * cfg.M * cfg.K * cfg.N)
			b.SetBytes(ops * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulF32(dst, a, mat, cfg.M, cfg.K, cfg.N)
			}
			b.ReportMetric(float64(ops)/1e9, "gops")
		})
	}
}

// BenchmarkMatMulBatchScaling tests how matmul scales with batch size
func BenchmarkMatMulBatchScaling(b *testing.B) {
	batchSizes := []int{1, 2, 4, 8, 16, 32, 64, 128}
	inDim := 512
	outDim := 2048

	for _, batch := range batchSizes {
		name := fmt.Sprintf("batch%d", batch)
		weight := make([]float32, outDim*inDim)
		input := make([]float32, batch*inDim)
		dst := make([]float32, batch*outDim)

		for i := range weight {
			weight[i] = float32(i%100) * 0.01
		}
		for i := range input {
			input[i] = float32(i%100) * 0.01
		}

		b.Run(name, func(b *testing.B) {
			b.ReportMetric(float64(batch), "batch")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulGGML(dst, weight, input, batch, inDim, outDim)
			}
		})
	}
}

// BenchmarkMatMulDimScaling tests how matmul scales with matrix dimensions
func BenchmarkMatMulDimScaling(b *testing.B) {
	dims := []int{128, 256, 512, 1024, 2048, 4096}
	batch := 4

	for _, dim := range dims {
		name := fmt.Sprintf("dim%d", dim)
		weight := make([]float32, dim*dim)
		input := make([]float32, batch*dim)
		dst := make([]float32, batch*dim)

		for i := range weight {
			weight[i] = float32(i%100) * 0.01
		}
		for i := range input {
			input[i] = float32(i%100) * 0.01
		}

		b.Run(name, func(b *testing.B) {
			b.ReportMetric(float64(dim), "dim")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulGGML(dst, weight, input, batch, dim, dim)
			}
		})
	}
}

// BenchmarkMatMulINT8 measures INT8 quantized matmul performance
func BenchmarkMatMulINT8(b *testing.B) {
	configs := []struct {
		name string
		M, K, N int
	}{
		{"Small_128x128x128", 128, 128, 128},
		{"Medium_256x512x256", 256, 512, 256},
		{"Large_512x2048x512", 512, 2048, 512},
		{"XLarge_1024x2048x1024", 1024, 2048, 1024},
	}

	for _, cfg := range configs {
		// Create quantized matrices
		aData := make([]float32, cfg.M*cfg.K)
		bData := make([]float32, cfg.K*cfg.N)
		for i := range aData {
			aData[i] = float32(i%100) * 0.01
		}
		for i := range bData {
			bData[i] = float32(i%100) * 0.01
		}

		A := QuantizeSymmetricINT8(aData, cfg.M, cfg.K)
		B := QuantizeSymmetricINT8(bData, cfg.K, cfg.N)
		dst := make([]float32, cfg.M*cfg.N)

		b.Run(cfg.name, func(b *testing.B) {
			ops := int64(2 * cfg.M * cfg.K * cfg.N)
			b.SetBytes(ops * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulINT8(&A, &B, dst)
			}
			b.ReportMetric(float64(ops)/1e9, "gops")
		})
	}
}

// BenchmarkMatMulINT8GGML measures GGML-style INT8 matmul
func BenchmarkMatMulINT8GGML(b *testing.B) {
	configs := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"Small_1x512x512", 1, 512, 512},
		{"Medium_4x512x2048", 4, 512, 2048},
		{"Large_8x2048x8192", 8, 2048, 8192},
	}

	for _, cfg := range configs {
		// Create quantized tensors
		weightData := make([]float32, cfg.outDim*cfg.inDim)
		inputData := make([]float32, cfg.batch*cfg.inDim)
		for i := range weightData {
			weightData[i] = float32(i%100) * 0.01
		}
		for i := range inputData {
			inputData[i] = float32(i%100) * 0.01
		}

		weight := QuantizeSymmetricINT8(weightData, cfg.outDim, cfg.inDim)
		input := QuantizeSymmetricINT8(inputData, cfg.batch, cfg.inDim)
		dst := make([]float32, cfg.batch*cfg.outDim)

		b.Run(cfg.name, func(b *testing.B) {
			ops := int64(2 * cfg.batch * cfg.inDim * cfg.outDim)
			b.SetBytes(ops * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulINT8GGML(dst, &weight, &input, cfg.batch, cfg.inDim, cfg.outDim)
			}
			b.ReportMetric(float64(ops)/1e9, "gops")
		})
	}
}

// BenchmarkMatMulFP32VsINT8 compares FP32 and INT8 matmul performance
func BenchmarkMatMulFP32VsINT8(b *testing.B) {
	batch := 4
	inDim := 512
	outDim := 2048

	// FP32 data
	weightFP32 := make([]float32, outDim*inDim)
	inputFP32 := make([]float32, batch*inDim)
	dstFP32 := make([]float32, batch*outDim)

	for i := range weightFP32 {
		weightFP32[i] = float32(i%100) * 0.01
	}
	for i := range inputFP32 {
		inputFP32[i] = float32(i%100) * 0.01
	}

	// INT8 data
	weightINT8 := QuantizeSymmetricINT8(weightFP32, outDim, inDim)
	inputINT8 := QuantizeSymmetricINT8(inputFP32, batch, inDim)
	dstINT8 := make([]float32, batch*outDim)

	b.Run("FP32", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MatMulGGML(dstFP32, weightFP32, inputFP32, batch, inDim, outDim)
		}
	})

	b.Run("INT8", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MatMulINT8GGML(dstINT8, &weightINT8, &inputINT8, batch, inDim, outDim)
		}
	})
}
