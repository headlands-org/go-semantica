package kernels

import (
	"math"
	"testing"
)

// TestMatMulGGMLSerial verifies serial version produces same output as parallel
func TestMatMulGGMLSerial(t *testing.T) {
	tests := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"small", 2, 64, 32},
		{"medium", 4, 128, 128},
		{"large", 8, 256, 512},
		{"large_output", 2, 128, 600}, // Tests outDim > parallelThreshold
		{"non_power_of_2", 3, 100, 75},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create random input data
			weight := make([]float32, tt.outDim*tt.inDim)
			input := make([]float32, tt.batch*tt.inDim)
			for i := range weight {
				weight[i] = float32(i%7) * 0.1
			}
			for i := range input {
				input[i] = float32(i%5) * 0.2
			}

			// Compute with parallel version
			dstParallel := make([]float32, tt.batch*tt.outDim)
			MatMulGGML(dstParallel, weight, input, tt.batch, tt.inDim, tt.outDim)

			// Compute with serial version
			dstSerial := make([]float32, tt.batch*tt.outDim)
			MatMulGGMLSerial(dstSerial, weight, input, tt.batch, tt.inDim, tt.outDim)

			// Compare results
			maxDiff := float32(0)
			for i := range dstParallel {
				diff := float32(math.Abs(float64(dstParallel[i] - dstSerial[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 1e-4 {
					t.Errorf("Mismatch at index %d: parallel=%f, serial=%f, diff=%f",
						i, dstParallel[i], dstSerial[i], diff)
					break
				}
			}

			t.Logf("Max difference: %e", maxDiff)
		})
	}
}

// TestMatMulQ8_0INT8Serial verifies serial version produces same output as parallel
func TestMatMulQ8_0INT8Serial(t *testing.T) {
	tests := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"small", 2, 64, 32},
		{"medium", 4, 128, 128},
		{"large", 8, 256, 512},
		{"non_multiple_of_32", 2, 100, 75},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create Q8_0 weight data (simplified for testing)
			blocksPerRow := (tt.inDim + 31) / 32
			bytesPerRow := blocksPerRow * 34
			weightData := make([]byte, tt.outDim*bytesPerRow)
			scales := make([]float32, tt.outDim*blocksPerRow)

			// Initialize weight data and scales
			for row := 0; row < tt.outDim; row++ {
				rowOffset := row * bytesPerRow
				for block := 0; block < blocksPerRow; block++ {
					blockOffset := rowOffset + block*34
					// Set scale (f16 = 0x3C00 = 1.0)
					weightData[blockOffset] = 0x00
					weightData[blockOffset+1] = 0x3C
					scales[row*blocksPerRow+block] = 1.0

					// Set quantized values
					for i := 0; i < 32 && block*32+i < tt.inDim; i++ {
						weightData[blockOffset+2+i] = byte((row*blocksPerRow + block + i) % 127)
					}
				}
			}

			// Create INT8 input
			inputData := make([]float32, tt.batch*tt.inDim)
			for i := range inputData {
				inputData[i] = float32(i%13) * 0.1
			}
			input := QuantizeSymmetricINT8(inputData, tt.batch, tt.inDim)

			// Compute with parallel version (via threshold)
			dstParallel := make([]float32, tt.batch*tt.outDim)
			MatMulQ8_0INT8(dstParallel, weightData, scales, &input, tt.batch, tt.inDim, tt.outDim, false)

			// Compute with serial version
			dstSerial := make([]float32, tt.batch*tt.outDim)
			MatMulQ8_0INT8Serial(dstSerial, weightData, scales, &input, tt.batch, tt.inDim, tt.outDim)

			// Compare results
			maxDiff := float32(0)
			for i := range dstParallel {
				diff := float32(math.Abs(float64(dstParallel[i] - dstSerial[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 1e-4 {
					t.Errorf("Mismatch at index %d: parallel=%f, serial=%f, diff=%f",
						i, dstParallel[i], dstSerial[i], diff)
					break
				}
			}

			t.Logf("Max difference: %e", maxDiff)
		})
	}
}

// TestMatMulQ8_0INT8FastSerial verifies Fast serial version produces same output as parallel
func TestMatMulQ8_0INT8FastSerial(t *testing.T) {
	tests := []struct {
		name   string
		batch  int
		inDim  int
		outDim int
	}{
		{"small", 2, 64, 32},
		{"medium", 4, 128, 128},
		{"large", 8, 256, 512},
		{"large_output", 2, 128, 600}, // Tests outDim > parallelThreshold
		{"non_multiple_of_32", 2, 100, 75},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create Q8_0 weight tensor
			blocksPerRow := (tt.inDim + 31) / 32
			qs := make([]int8, tt.outDim*tt.inDim)
			scales := make([]float32, tt.outDim*blocksPerRow)

			// Initialize weight data
			for row := 0; row < tt.outDim; row++ {
				for col := 0; col < tt.inDim; col++ {
					qs[row*tt.inDim+col] = int8((row + col) % 127)
				}
				for block := 0; block < blocksPerRow; block++ {
					scales[row*blocksPerRow+block] = 1.0 / 127.0
				}
			}

			weight := &Q8_0Tensor{
				Qs:     qs,
				Scales: scales,
				Rows:   tt.outDim,
				Cols:   tt.inDim,
			}

			// Create INT8 input
			inputData := make([]float32, tt.batch*tt.inDim)
			for i := range inputData {
				inputData[i] = float32(i%13) * 0.1
			}
			input := QuantizeSymmetricINT8(inputData, tt.batch, tt.inDim)

			// Compute with parallel/auto version
			dstAuto := make([]float32, tt.batch*tt.outDim)
			MatMulQ8_0INT8Fast(dstAuto, weight, &input, tt.batch, tt.inDim, tt.outDim)

			// Compute with serial version
			dstSerial := make([]float32, tt.batch*tt.outDim)
			MatMulQ8_0INT8FastSerial(dstSerial, weight, &input, tt.batch, tt.inDim, tt.outDim)

			// Compare results
			maxDiff := float32(0)
			for i := range dstAuto {
				diff := float32(math.Abs(float64(dstAuto[i] - dstSerial[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 1e-4 {
					t.Errorf("Mismatch at index %d: auto=%f, serial=%f, diff=%f",
						i, dstAuto[i], dstSerial[i], diff)
					break
				}
			}

			t.Logf("Max difference: %e", maxDiff)
		})
	}
}

// BenchmarkMatMulGGMLSerial benchmarks the serial version
func BenchmarkMatMulGGMLSerial(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i % 7)
	}
	for i := range input {
		input[i] = float32(i % 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulGGMLSerial(dst, weight, input, batch, inDim, outDim)
	}
}

// BenchmarkMatMulGGMLParallel benchmarks the parallel version
func BenchmarkMatMulGGMLParallel(b *testing.B) {
	// Force parallel by using large outDim (>= 256)
	batch, inDim, outDim := 4, 256, 512
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i % 7)
	}
	for i := range input {
		input[i] = float32(i % 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulGGML(dst, weight, input, batch, inDim, outDim)
	}
}

// BenchmarkMatMulQ8_0INT8FastSerial benchmarks the Fast serial version
func BenchmarkMatMulQ8_0INT8FastSerial(b *testing.B) {
	batch, inDim, outDim := 4, 256, 256
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
		MatMulQ8_0INT8FastSerial(dst, weight, &input, batch, inDim, outDim)
	}
}

// BenchmarkMatMulGGMLSerialSmall benchmarks serial for small matrices
func BenchmarkMatMulGGMLSerialSmall(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)
	dst := make([]float32, batch*outDim)

	for i := range weight {
		weight[i] = float32(i % 7)
	}
	for i := range input {
		input[i] = float32(i % 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulGGMLSerial(dst, weight, input, batch, inDim, outDim)
	}
}

// BenchmarkCompareSerialVsAutoSmall compares serial vs auto for small matrices
func BenchmarkCompareSerialVsAutoSmall(b *testing.B) {
	batch, inDim, outDim := 1, 128, 128
	weight := make([]float32, outDim*inDim)
	input := make([]float32, batch*inDim)

	for i := range weight {
		weight[i] = float32(i % 7)
	}
	for i := range input {
		input[i] = float32(i % 5)
	}

	b.Run("Serial", func(b *testing.B) {
		dst := make([]float32, batch*outDim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MatMulGGMLSerial(dst, weight, input, batch, inDim, outDim)
		}
	})

	b.Run("Auto", func(b *testing.B) {
		dst := make([]float32, batch*outDim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MatMulGGML(dst, weight, input, batch, inDim, outDim)
		}
	})
}
