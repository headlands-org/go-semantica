package kernels

import (
	"unsafe"
)

// QuantizedTensorINT8 represents a symmetric INT8-quantized tensor
// Values are quantized as: q = round(x / scale)
// Dequantized as: x = q * scale
type QuantizedTensorINT8 struct {
	Data  []int8  // Quantized values [-127, 127]
	Scale float32 // Scaling factor
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

// QuantizeSymmetricINT8Into quantizes the given tensor into a reusable destination buffer.
func QuantizeSymmetricINT8Into(dst *QuantizedTensorINT8, x []float32, rows, cols int) {
	if len(x) != rows*cols {
		panic("QuantizeSymmetricINT8Into: size mismatch")
	}

	absMax := findAbsMax(x)

	if absMax == 0 {
		if cap(dst.Data) < len(x) {
			dst.Data = make([]int8, len(x))
		}
		dst.Data = dst.Data[:len(x)]
		for i := range dst.Data {
			dst.Data[i] = 0
		}
		dst.Scale = 1.0
		dst.Rows = rows
		dst.Cols = cols
		return
	}

	scale := absMax / 127.0
	invScale := float32(0)
	if scale != 0 {
		invScale = 1.0 / scale
	}
	if cap(dst.Data) < len(x) {
		dst.Data = make([]int8, len(x))
	}
	data := dst.Data[:len(x)]

	i := 0
	n := len(x)
	for ; i+8 <= n; i += 8 {
		for j := 0; j < 8; j++ {
			v := x[i+j]
			scaled := v * invScale
			var q int32
			if scaled >= 0 {
				q = int32(scaled + 0.5)
			} else {
				q = int32(scaled - 0.5)
			}
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			data[i+j] = int8(q)
		}
	}
	for ; i < n; i++ {
		scaled := x[i] * invScale
		var q int32
		if scaled >= 0 {
			q = int32(scaled + 0.5)
		} else {
			q = int32(scaled - 0.5)
		}
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		data[i] = int8(q)
	}

	dst.Data = data
	dst.Scale = scale
	dst.Rows = rows
	dst.Cols = cols
}

// QuantizeSymmetricINT8 quantizes FP32 tensor to INT8 using symmetric quantization
// scale = max(abs(x)) / 127.0
// This preserves zero exactly (0.0 -> 0)
func QuantizeSymmetricINT8(x []float32, rows, cols int) QuantizedTensorINT8 {
	var dst QuantizedTensorINT8
	QuantizeSymmetricINT8Into(&dst, x, rows, cols)
	return dst
}

func findAbsMax(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}

	max0 := float32(0)
	max1 := float32(0)
	max2 := float32(0)
	max3 := float32(0)

	i := 0
	n := len(x)
	for ; i+4 <= n; i += 4 {
		v0 := x[i+0]
		if v0 < 0 {
			v0 = -v0
		}
		if v0 > max0 {
			max0 = v0
		}
		v1 := x[i+1]
		if v1 < 0 {
			v1 = -v1
		}
		if v1 > max1 {
			max1 = v1
		}
		v2 := x[i+2]
		if v2 < 0 {
			v2 = -v2
		}
		if v2 > max2 {
			max2 = v2
		}
		v3 := x[i+3]
		if v3 < 0 {
			v3 = -v3
		}
		if v3 > max3 {
			max3 = v3
		}
	}

	for ; i < n; i++ {
		v := x[i]
		if v < 0 {
			v = -v
		}
		if v > max0 {
			max0 = v
		}
	}

	if max1 > max0 {
		max0 = max1
	}
	if max2 > max0 {
		max0 = max2
	}
	if max3 > max0 {
		max0 = max3
	}
	return max0
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

// ParseQ8_0Tensor converts raw Q8_0 bytes into a tensor with pre-parsed scales and values.
func ParseQ8_0Tensor(data []byte, rows, cols int) *Q8_0Tensor {
	if rows <= 0 || cols <= 0 {
		panic("ParseQ8_0Tensor: invalid shape")
	}

	blocksPerRow := (cols + 31) / 32
	expectedBytes := rows * blocksPerRow * 34
	if len(data) < expectedBytes {
		panic("ParseQ8_0Tensor: data too small")
	}

	qs := make([]int8, rows*cols)
	scales := make([]float32, rows*blocksPerRow)

	for r := 0; r < rows; r++ {
		rowOffset := r * blocksPerRow * 34
		qsBase := r * cols
		scaleBase := r * blocksPerRow

		for b := 0; b < blocksPerRow; b++ {
			blockOffset := rowOffset + b*34
			scaleBytes := uint16(data[blockOffset]) | uint16(data[blockOffset+1])<<8
			scales[scaleBase+b] = float16ToFloat32(scaleBytes)

			blockStart := b * 32
			remaining := cols - blockStart
			if remaining <= 0 {
				continue
			}
			blockSize := 32
			if remaining < 32 {
				blockSize = remaining
			}

			qBytes := data[blockOffset+2 : blockOffset+2+blockSize]
			qsRow := qs[qsBase+blockStart : qsBase+blockStart+blockSize]
			for i := 0; i < blockSize; i++ {
				qsRow[i] = int8(qBytes[i])
			}
		}
	}

	return &Q8_0Tensor{
		Qs:     qs,
		Scales: scales,
		Rows:   rows,
		Cols:   cols,
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
			// Weight row j starts at offset j * bytesPerRow
			weightRowOffset := j * bytesPerRow
			scaleBaseIdx := j * blocksPerRow

			// Use fully-optimized assembly inner loop for full blocks
			// This keeps all accumulators in YMM registers with vectorized FMA
			sum := float32(0)
			if fullBlocks > 0 {
				inputPtr := (*int8)(unsafe.Pointer(&input.Data[inputOffset]))
				weightPtr := (*byte)(unsafe.Pointer(&weightData[weightRowOffset]))
				scalesPtr := (*float32)(unsafe.Pointer(&scales[scaleBaseIdx]))
				sum = matmulInnerLoop(inputPtr, weightPtr, scalesPtr, fullBlocks)
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

			// Apply inputScale once at the end
			dst[i*outDim+j] = sum * inputScale
		}
	}
}
