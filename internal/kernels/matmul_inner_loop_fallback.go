//go:build (!amd64 && !arm64) || noasm

package kernels

import "unsafe"

const (
	blockSizeBytes   = 32
	blockStrideBytes = 34 // 2-byte float16 scale + 32 quantized values
)

// matmulInnerLoopAsm is the portable fallback used on non-AMD64 targets.
// The AMD implementation lives in assembly; other architectures reuse the
// shared INT8 dot product helper so that we still get the best available SIMD
// while keeping the code easy to reason about (and correct) on every target.
func matmulInnerLoopAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32 {
	if numBlocks <= 0 {
		return 0
	}

	inputSlice := unsafe.Slice(inputRow, numBlocks*blockSizeBytes)
	weightBytes := unsafe.Slice(weightData, numBlocks*blockStrideBytes)
	scaleSlice := unsafe.Slice(scales, numBlocks)

	sum := float32(0)
	for block := 0; block < numBlocks; block++ {
		blockStart := block * blockSizeBytes
		inputBlock := inputSlice[blockStart : blockStart+blockSizeBytes]

		qsOffset := block*blockStrideBytes + 2 // skip float16 scale in GGUF block layout
		weightBlockPtr := (*int8)(unsafe.Pointer(&weightBytes[qsOffset]))
		weightBlock := unsafe.Slice(weightBlockPtr, blockSizeBytes)

		acc := dotProductINT8SIMD(inputBlock, weightBlock, blockSizeBytes)
		sum += float32(acc) * scaleSlice[block]
	}

	return sum
}

func matmulInnerLoop(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32 {
	return matmulInnerLoopAsm(inputRow, weightData, scales, numBlocks)
}
