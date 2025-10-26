package kernels

import (
	"fmt"
	"testing"
)

// Benchmark cache-aware block sizes for FP32 matmul serial implementation
// Tests block sizes: 16, 32, 64, 128 to find optimal L1/L2 cache utilization

// Test matrix dimensions representing typical workloads:
// - Small: embedding layer (batch=1, inDim=128, outDim=128)
// - Medium: attention projection (batch=4, inDim=256, outDim=256)
// - Large: FFN layers (batch=4, inDim=512, outDim=2048)

// benchmarkMatMulGGMLSerialWithBlockSize runs matmul with a custom block size
func benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input []float32, batch, inDim, outDim, blockSize int) {
	// Zero output
	for i := range dst[:batch*outDim] {
		dst[i] = 0
	}

	for i0 := 0; i0 < batch; i0 += blockSize {
		i1 := min(i0+blockSize, batch)
		for j0 := 0; j0 < outDim; j0 += blockSize {
			j1 := min(j0+blockSize, outDim)
			for k0 := 0; k0 < inDim; k0 += blockSize {
				k1 := min(k0+blockSize, inDim)

				// Inner loops with SIMD acceleration
				for i := i0; i < i1; i++ {
					inputBase := i * inDim
					dstBase := i * outDim

					for j := j0; j < j1; j++ {
						weightBase := j * inDim
						blockLen := k1 - k0

						// Use SIMD dot product for the block
						sum := dotProductSIMD(
							input[inputBase+k0:inputBase+k1],
							weight[weightBase+k0:weightBase+k1],
							blockLen,
						)

						dst[dstBase+j] += sum
					}
				}
			}
		}
	}
}

// benchmarkMatMulQ8_0INT8FastSerialWithBlockSize runs Q8_0 INT8 matmul with custom block size
func benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst []float32, weight *Q8_0Tensor, input *QuantizedTensorINT8, batch, inDim, outDim, blockSize int) {
	blocksPerRow := (inDim + 31) / 32

	// Zero output
	for i := range dst[:batch*outDim] {
		dst[i] = 0
	}

	for i0 := 0; i0 < batch; i0 += blockSize {
		i1 := min(i0+blockSize, batch)
		for j0 := 0; j0 < outDim; j0 += blockSize {
			j1 := min(j0+blockSize, outDim)

			for i := i0; i < i1; i++ {
				for j := j0; j < j1; j++ {
					sum := float32(0)

					// Process weight row j in blocks of 32
					for blockIdx := 0; blockIdx < blocksPerRow; blockIdx++ {
						// Get scale for this block (pre-converted to float32)
						scaleIdx := j*blocksPerRow + blockIdx
						scale := weight.Scales[scaleIdx]

						// Block boundaries
						blockStart := blockIdx * 32
						blockEnd := blockStart + 32
						if blockEnd > inDim {
							blockEnd = inDim
						}
						blockSize := blockEnd - blockStart

						// Dot product with SIMD
						inputSlice := input.Data[i*inDim+blockStart : i*inDim+blockStart+blockSize]
						weightSlice := weight.Qs[j*inDim+blockStart : j*inDim+blockStart+blockSize]
						blockSum := dotProductINT8SIMD(inputSlice, weightSlice, blockSize)

						// Dequantize: INT32 accumulator * Q8_0 scale * input scale
						sum += float32(blockSum) * scale * input.Scale
					}

					dst[i*outDim+j] = sum
				}
			}
		}
	}
}

// =============================================================================
// FP32 MatMul Block Size Benchmarks - Small Matrices (128x128)
// =============================================================================

func BenchmarkMatMulGGMLSerial_Small_Block16(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 16)
	}
}

func BenchmarkMatMulGGMLSerial_Small_Block32(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 32)
	}
}

func BenchmarkMatMulGGMLSerial_Small_Block64(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 64)
	}
}

func BenchmarkMatMulGGMLSerial_Small_Block128(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 128)
	}
}

// =============================================================================
// FP32 MatMul Block Size Benchmarks - Medium Matrices (256x256, batch=4)
// =============================================================================

func BenchmarkMatMulGGMLSerial_Medium_Block16(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 16)
	}
}

func BenchmarkMatMulGGMLSerial_Medium_Block32(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 32)
	}
}

func BenchmarkMatMulGGMLSerial_Medium_Block64(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 64)
	}
}

func BenchmarkMatMulGGMLSerial_Medium_Block128(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 128)
	}
}

// =============================================================================
// FP32 MatMul Block Size Benchmarks - Large Matrices (512x2048, batch=4)
// =============================================================================

func BenchmarkMatMulGGMLSerial_Large_Block16(b *testing.B) {
	batch, inDim, outDim := 4, 512, 2048
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 16)
	}
}

func BenchmarkMatMulGGMLSerial_Large_Block32(b *testing.B) {
	batch, inDim, outDim := 4, 512, 2048
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 32)
	}
}

func BenchmarkMatMulGGMLSerial_Large_Block64(b *testing.B) {
	batch, inDim, outDim := 4, 512, 2048
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 64)
	}
}

func BenchmarkMatMulGGMLSerial_Large_Block128(b *testing.B) {
	batch, inDim, outDim := 4, 512, 2048
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, 128)
	}
}

// =============================================================================
// Q8_0 INT8 MatMul Block Size Benchmarks - Small Matrices (128x128)
// =============================================================================

func BenchmarkMatMulQ8INT8Fast_Small_Block16(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	blocksPerRow := (inDim + 31) / 32

	// Create weight
	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	// Create input
	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 16)
	}
}

func BenchmarkMatMulQ8INT8Fast_Small_Block32(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 32)
	}
}

func BenchmarkMatMulQ8INT8Fast_Small_Block64(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 64)
	}
}

func BenchmarkMatMulQ8INT8Fast_Small_Block128(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 128)
	}
}

// =============================================================================
// Q8_0 INT8 MatMul Block Size Benchmarks - Medium Matrices (256x256, batch=4)
// =============================================================================

func BenchmarkMatMulQ8INT8Fast_Medium_Block16(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 16)
	}
}

func BenchmarkMatMulQ8INT8Fast_Medium_Block32(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 32)
	}
}

func BenchmarkMatMulQ8INT8Fast_Medium_Block64(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 64)
	}
}

func BenchmarkMatMulQ8INT8Fast_Medium_Block128(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	blocksPerRow := (inDim + 31) / 32

	qs := make([]int8, outDim*inDim)
	scales := make([]float32, outDim*blocksPerRow)
	for i := range qs {
		qs[i] = int8(i % 127)
	}
	for i := range scales {
		scales[i] = 1.0 / 127.0
	}
	weight := &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   outDim,
		Cols:   inDim,
	}

	inputData := make([]float32, batch*inDim)
	for i := range inputData {
		inputData[i] = float32(i % 13)
	}
	input := QuantizeSymmetricINT8(inputData, batch, inDim)

	dst := make([]float32, batch*outDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkMatMulQ8_0INT8FastSerialWithBlockSize(dst, weight, &input, batch, inDim, outDim, 128)
	}
}

// =============================================================================
// Helper benchmark to measure current implementation performance
// =============================================================================

func BenchmarkMatMulGGMLSerial_Current(b *testing.B) {
	tests := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"Small_128x128", 1, 128, 128},
		{"Medium_256x256_batch4", 4, 256, 256},
		{"Large_512x2048_batch4", 4, 512, 2048},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			weight := make([]float32, tt.outDim*tt.inDim)
			input := make([]float32, tt.batch*tt.inDim)
			dst := make([]float32, tt.batch*tt.outDim)

			for i := range weight {
				weight[i] = float32(i%7) * 0.1
			}
			for i := range input {
				input[i] = float32(i%5) * 0.2
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Use internal serial function directly
				matMulGGMLSerial(dst, weight, input, tt.batch, tt.inDim, tt.outDim)
			}
		})
	}
}

func BenchmarkMatMulQ8INT8FastSerial_Current(b *testing.B) {
	tests := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"Small_128x128", 1, 128, 128},
		{"Medium_256x256_batch4", 4, 256, 256},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			blocksPerRow := (tt.inDim + 31) / 32

			qs := make([]int8, tt.outDim*tt.inDim)
			scales := make([]float32, tt.outDim*blocksPerRow)
			for i := range qs {
				qs[i] = int8(i % 127)
			}
			for i := range scales {
				scales[i] = 1.0 / 127.0
			}
			weight := &Q8_0Tensor{
				Qs:     qs,
				Scales: scales,
				Rows:   tt.outDim,
				Cols:   tt.inDim,
			}

			inputData := make([]float32, tt.batch*tt.inDim)
			for i := range inputData {
				inputData[i] = float32(i % 13)
			}
			input := QuantizeSymmetricINT8(inputData, tt.batch, tt.inDim)

			dst := make([]float32, tt.batch*tt.outDim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Use internal serial function directly
				matMulQ8_0INT8FastSerial(dst, weight, &input, tt.batch, tt.inDim, tt.outDim)
			}
		})
	}
}

// =============================================================================
// Comparison benchmark - run this to generate summary
// =============================================================================

func BenchmarkBlockSizeComparison(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	blockSizes := []int{16, 32, 64, 128}

	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i%7) * 0.1
	}
	for i := range input {
		input[i] = float32(i%5) * 0.2
	}

	for _, blockSize := range blockSizes {
		b.Run(fmt.Sprintf("FP32_Block%d", blockSize), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchmarkMatMulGGMLSerialWithBlockSize(dst, weight, input, batch, inDim, outDim, blockSize)
			}
		})
	}
}
