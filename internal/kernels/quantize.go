package kernels

import (
	"math"
	"unsafe"
)

// QuantizedTensorINT8 represents a symmetric INT8-quantized tensor
// Values are quantized as: q = round(x / scale)
// Dequantized as: x = q * scale
type QuantizedTensorINT8 struct {
	Data  []int8   // Quantized values [-127, 127]
	Scale float32  // Scaling factor
	// For matrix ops: shape info
	Rows int
	Cols int
}

// Q8_0Tensor represents pre-parsed Q8_0 data for fast inference
// Avoids repeated float16->float32 conversion during matmul
type Q8_0Tensor struct {
	Qs     []int8    // All quantized values (flattened)
	Scales []float32 // Pre-converted float32 scales (one per 32-element block)
	Rows   int
	Cols   int
}

// QuantizeSymmetricINT8 quantizes FP32 tensor to INT8 using symmetric quantization
// scale = max(abs(x)) / 127.0
// This preserves zero exactly (0.0 -> 0)
func QuantizeSymmetricINT8(x []float32, rows, cols int) QuantizedTensorINT8 {
	if len(x) != rows*cols {
		panic("QuantizeSymmetricINT8: size mismatch")
	}

	// Find absolute maximum for scaling
	absMax := float32(0)
	for _, v := range x {
		if abs := math.Abs(float64(v)); float32(abs) > absMax {
			absMax = float32(abs)
		}
	}

	// Avoid division by zero
	if absMax == 0 {
		return QuantizedTensorINT8{
			Data:  make([]int8, len(x)),
			Scale: 1.0,
			Rows:  rows,
			Cols:  cols,
		}
	}

	// Scale factor to map [-absMax, absMax] to [-127, 127]
	scale := absMax / 127.0

	// Quantize
	data := make([]int8, len(x))
	for i, v := range x {
		// Round to nearest integer
		quantized := math.Round(float64(v / scale))
		// Clamp to INT8 range
		if quantized > 127 {
			quantized = 127
		} else if quantized < -127 {
			quantized = -127
		}
		data[i] = int8(quantized)
	}

	return QuantizedTensorINT8{
		Data:  data,
		Scale: scale,
		Rows:  rows,
		Cols:  cols,
	}
}

// Dequantize converts INT8 back to FP32
func (q *QuantizedTensorINT8) Dequantize() []float32 {
	result := make([]float32, len(q.Data))
	for i, v := range q.Data {
		result[i] = float32(v) * q.Scale
	}
	return result
}

// DequantizeTo dequantizes into existing buffer (avoids allocation)
func (q *QuantizedTensorINT8) DequantizeTo(dst []float32) {
	if len(dst) < len(q.Data) {
		panic("DequantizeTo: dst too small")
	}
	for i, v := range q.Data {
		dst[i] = float32(v) * q.Scale
	}
}

// MatMulINT8 performs matrix multiplication with INT8 quantized tensors
// A: [M, K] INT8
// B: [K, N] INT8
// C: [M, N] FP32 (output is dequantized)
//
// Algorithm:
// 1. Multiply INT8 values -> accumulate in INT32 (avoid overflow)
// 2. Scale by A.scale * B.scale
// 3. Optionally requantize or return FP32
func MatMulINT8(A, B *QuantizedTensorINT8, dst []float32) {
	M := A.Rows
	K := A.Cols
	N := B.Cols

	if B.Rows != K {
		panic("MatMulINT8: dimension mismatch")
	}
	if len(dst) < M*N {
		panic("MatMulINT8: dst too small")
	}

	// Combined scale factor
	combinedScale := A.Scale * B.Scale

	// Zero output
	for i := range dst[:M*N] {
		dst[i] = 0
	}

	// INT8 multiplication with INT32 accumulation
	// This is the scalar version - will optimize with SIMD later
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := int32(0) // INT32 accumulator to avoid overflow

			// Dot product
			for k := 0; k < K; k++ {
				aVal := int32(A.Data[i*K+k])
				bVal := int32(B.Data[k*N+j])
				sum += aVal * bVal
			}

			// Dequantize: convert INT32 -> FP32 with combined scale
			dst[i*N+j] = float32(sum) * combinedScale
		}
	}
}

// MatMulINT8GGML is the GGML-compatible version (weight-first semantics)
// weight: [outDim, inDim] INT8
// input: [batch, inDim] INT8
// output: [batch, outDim] FP32
func MatMulINT8GGML(dst []float32, weight, input *QuantizedTensorINT8, batch, inDim, outDim int) {
	if weight.Rows != outDim || weight.Cols != inDim {
		panic("MatMulINT8GGML: weight shape mismatch")
	}
	if input.Rows != batch || input.Cols != inDim {
		panic("MatMulINT8GGML: input shape mismatch")
	}
	if len(dst) < batch*outDim {
		panic("MatMulINT8GGML: dst too small")
	}

	combinedScale := weight.Scale * input.Scale

	// Zero output
	for i := range dst[:batch*outDim] {
		dst[i] = 0
	}

	// output[i, j] = sum_k input[i, k] * weight[j, k]
	for i := 0; i < batch; i++ {
		for j := 0; j < outDim; j++ {
			sum := int32(0)

			for k := 0; k < inDim; k++ {
				inputVal := int32(input.Data[i*inDim+k])
				weightVal := int32(weight.Data[j*inDim+k])
				sum += inputVal * weightVal
			}

					dst[i*outDim+j] = float32(sum) * combinedScale
		}
	}
}

// MatMulQ8_0INT8Fast performs matrix multiplication with pre-parsed Q8_0 weights and INT8 activations
// This is the optimized version that avoids repeated Q8_0 block parsing
//
// weight: Pre-parsed Q8_0 quantized weight tensor
// input: INT8 quantized activations
// dst: FP32 output (dequantized)
//
// Weight is [outDim, inDim] in row-major order
// Input is [batch, inDim]
// Output is [batch, outDim]
func MatMulQ8_0INT8Fast(dst []float32, weight *Q8_0Tensor, input *QuantizedTensorINT8, batch, inDim, outDim int) {
	if weight.Rows != outDim || weight.Cols != inDim {
		panic("MatMulQ8_0INT8Fast: weight shape mismatch")
	}
	if input.Rows != batch || input.Cols != inDim {
		panic("MatMulQ8_0INT8Fast: input shape mismatch")
	}
	if len(dst) < batch*outDim {
		panic("MatMulQ8_0INT8Fast: dst too small")
	}

	// Zero output
	for i := range dst[:batch*outDim] {
		dst[i] = 0
	}

	// Always use serial execution (simpler, avoids WaitGroup contention)
	matMulQ8_0INT8FastSerial(dst, weight, input, batch, inDim, outDim)
}

// matMulQ8_0INT8FastSerial is the serial implementation
func matMulQ8_0INT8FastSerial(dst []float32, weight *Q8_0Tensor, input *QuantizedTensorINT8, batch, inDim, outDim int) {
	blocksPerRow := (inDim + 31) / 32
	inputScale := input.Scale

	for i := 0; i < batch; i++ {
		inputOffset := i * inDim

		for j := 0; j < outDim; j++ {
			sum := float32(0)
			weightOffset := j * inDim
			scaleBaseIdx := j * blocksPerRow

			// Process weight row j in blocks of 32
			// OPTIMIZATION: Hoist inputScale multiplication outside block loop
			for blockIdx := 0; blockIdx < blocksPerRow; blockIdx++ {
				// Get scale for this block (pre-converted to float32)
				scale := weight.Scales[scaleBaseIdx+blockIdx]

				// Block boundaries
				blockStart := blockIdx * 32
				blockEnd := blockStart + 32
				if blockEnd > inDim {
					blockEnd = inDim
				}
				blockSize := blockEnd - blockStart

				// Dot product with SIMD
				inputSlice := input.Data[inputOffset+blockStart : inputOffset+blockStart+blockSize]
				weightSlice := weight.Qs[weightOffset+blockStart : weightOffset+blockStart+blockSize]
				blockSum := dotProductINT8SIMD(inputSlice, weightSlice, blockSize)

				// Accumulate without inputScale (apply once at end)
				sum += float32(blockSum) * scale
			}

			// Apply inputScale once at the end instead of per-block
			dst[i*outDim+j] = sum * inputScale
		}
	}
}

// MatMulQ8_0INT8 performs matrix multiplication with Q8_0 weights and INT8 activations
// This preserves per-block quantization from Q8_0 for better accuracy
//
// weightData: Q8_0 quantized weight tensor (raw bytes from GGUF)
// scales: Pre-extracted float32 scales (one per 32-element block)
// input: INT8 quantized activations
// dst: FP32 output (dequantized)
//
// Weight is [outDim, inDim] in row-major order
// Input is [batch, inDim]
// Output is [batch, outDim]
//
// DEPRECATED: Use MatMulQ8_0INT8Fast with pre-parsed weights for better performance
func MatMulQ8_0INT8(dst []float32, weightData []byte, scales []float32, input *QuantizedTensorINT8, batch, inDim, outDim int) {
	if input.Rows != batch || input.Cols != inDim {
		panic("MatMulQ8_0INT8: input shape mismatch")
	}
	if len(dst) < batch*outDim {
		panic("MatMulQ8_0INT8: dst too small")
	}

	// Q8_0 format: blocks of 32 int8 values + 1 float16 scale (34 bytes per block)
	blocksPerRow := (inDim + 31) / 32
	bytesPerRow := blocksPerRow * 34

	expectedBytes := outDim * bytesPerRow
	if len(weightData) < expectedBytes {
		panic("MatMulQ8_0INT8: weightData too small")
	}

	// Zero output
	for i := range dst[:batch*outDim] {
		dst[i] = 0
	}

	// Use serial execution (simpler, avoids WaitGroup contention when called from parallel contexts)
	matMulQ8_0INT8Serial(dst, weightData, scales, input, batch, inDim, outDim, blocksPerRow, bytesPerRow)
}

// matMulQ8_0INT8Serial is the serial implementation
func matMulQ8_0INT8Serial(dst []float32, weightData []byte, scales []float32, input *QuantizedTensorINT8, batch, inDim, outDim, blocksPerRow, bytesPerRow int) {
	// Hoist constant values
	inputScale := input.Scale
	fullBlocks := inDim / 32
	hasPartialBlock := inDim%32 != 0

	// For each output position
	for i := 0; i < batch; i++ {
		inputOffset := i * inDim

		for j := 0; j < outDim; j++ {
			sum := float32(0)

			// Weight row j starts at offset j * bytesPerRow
			weightRowOffset := j * bytesPerRow
			scaleBaseIdx := j * blocksPerRow

			// Process full blocks of 32 (fast path)
			// OPTIMIZATION: Hoist inputScale multiplication outside block loop
			for blockIdx := 0; blockIdx < fullBlocks; blockIdx++ {
				blockOffset := weightRowOffset + blockIdx*34
				scale := scales[scaleBaseIdx+blockIdx]
				qsOffset := blockOffset + 2
				blockStart := blockIdx * 32

				// Direct assembly call - bypass dispatcher overhead
				inputPtr := (*int8)(unsafe.Pointer(&input.Data[inputOffset+blockStart]))
				weightPtr := (*int8)(unsafe.Pointer(&weightData[qsOffset]))
				blockSum := dotProductINT8Asm(inputPtr, weightPtr, 32)

				// Accumulate without inputScale (apply once at end)
				sum += float32(blockSum) * scale
			}

			// Handle partial block if needed
			if hasPartialBlock {
				blockIdx := fullBlocks
				blockOffset := weightRowOffset + blockIdx*34
				scale := scales[scaleBaseIdx+blockIdx]
				qsOffset := blockOffset + 2
				blockStart := blockIdx * 32
				blockSize := inDim - blockStart

				inputSlice := input.Data[inputOffset+blockStart : inputOffset+blockStart+blockSize]
				weightSlice := unsafe.Slice((*int8)(unsafe.Pointer(&weightData[qsOffset])), blockSize)
				blockSum := dotProductINT8SIMD(inputSlice, weightSlice, blockSize)

				sum += float32(blockSum) * scale
			}

			// Apply inputScale once at the end instead of per-block
			dst[i*outDim+j] = sum * inputScale
		}
	}
}

