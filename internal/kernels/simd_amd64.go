//go:build !noasm
// +build !noasm

package kernels

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// SIMD support flags
var (
	hasAVX2       = cpu.X86.HasAVX2
	hasAVX512     = cpu.X86.HasAVX512F
	hasAVX512VNNI = cpu.X86.HasAVX512VNNI
)

// dotProductAVX2Asm is the assembly implementation
// Processes 8 float32 values per iteration using AVX2
//
//go:noescape
func dotProductAVX2Asm(a, b *float32, n int) float32

// dotProductINT8VNNI is the AVX512 VNNI assembly implementation for INT8
// Processes 16 int8 values per iteration using VPDPBUSD
// Returns int32 accumulator (caller handles scaling)
//
//go:noescape
func dotProductINT8VNNI(a, b *int8, n int) int32

// matmulInnerLoopAsm processes one output element with fully-optimized vectorized FMA
// This is the performance-critical loop that keeps all accumulators in YMM registers
//
//go:noescape
func matmulInnerLoopAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32

type matmulInnerLoopFn func(*int8, *byte, *float32, int) float32

var currentMatmulInnerLoop matmulInnerLoopFn = matmulInnerLoopAsm

func init() {
	if hasAVX512VNNI {
		currentMatmulInnerLoop = matmulInnerLoopVNNI
	}
}

func matmulInnerLoop(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32 {
	return currentMatmulInnerLoop(inputRow, weightData, scales, numBlocks)
}

func matmulInnerLoopVNNI(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32 {
	const blockSize = 32
	const blockBytes = 34

	var acc float32
	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		scale := *(*float32)(unsafe.Add(unsafe.Pointer(scales), uintptr(blockIdx)*4))
		inputPtr := (*int8)(unsafe.Add(unsafe.Pointer(inputRow), uintptr(blockIdx*blockSize)))
		weightPtr := (*int8)(unsafe.Add(unsafe.Pointer(weightData), uintptr(blockIdx*blockBytes+2)))

		dot := dotProductINT8Asm(inputPtr, weightPtr, blockSize)
		acc += float32(dot) * scale
	}
	return acc
}

// dotProductINT8Asm is the direct assembly call without dispatcher overhead
// IMPORTANT: Caller must ensure slices are valid and n >= 16 for SIMD
func dotProductINT8Asm(a, b *int8, n int) int32 {
	// Direct call to AVX512 VNNI - no checks, no branches
	return dotProductINT8VNNI(a, b, n)
}

// dotProductSIMD is the SIMD-accelerated version with runtime feature detection
func dotProductSIMD(a, b []float32, n int) float32 {
	if n == 0 {
		return 0
	}

	// Ensure slices are large enough
	if len(a) < n || len(b) < n {
		panic("dotProductSIMD: slice too small")
	}

	// Use AVX2 if available (processes 8 floats at a time)
	if hasAVX2 && n >= 8 {
		return dotProductAVX2Asm(&a[0], &b[0], n)
	}

	// Fallback to scalar
	return dotProductScalar(a, b, n)
}

// dotProductScalar is the fallback scalar implementation
func dotProductScalar(a, b []float32, n int) float32 {
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// dotProductINT8SIMD is the SIMD-accelerated INT8 dot product
// Returns int32 accumulator (caller applies scaling)
//
//go:inline
func dotProductINT8SIMD(a, b []int8, n int) int32 {
	if n == 0 {
		return 0
	}

	// Ensure slices are large enough
	if len(a) < n || len(b) < n {
		panic("dotProductINT8SIMD: slice too small")
	}

	// Use AVX512 VNNI if available (processes 16 int8s at a time)
	if hasAVX512VNNI && n >= 16 {
		return dotProductINT8VNNI(&a[0], &b[0], n)
	}

	// Fallback to scalar INT32 accumulation
	sum := int32(0)
	for i := 0; i < n; i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}

// vecMulF32SIMD is the assembly implementation for element-wise multiplication
// dst[i] = a[i] * b[i]
// Processes 8 float32 values per iteration using AVX2
//
//go:noescape
func vecMulF32SIMD(dst, a, b *float32, n int)

// vecAddF32SIMD is the assembly implementation for element-wise addition
// dst[i] = a[i] + b[i]
// Processes 8 float32 values per iteration using AVX2
//
//go:noescape
func vecAddF32SIMD(dst, a, b *float32, n int)

// vecScaleF32SIMD is the assembly implementation for vector scaling
// dst[i] = a[i] * scale
// Processes 8 float32 values per iteration using AVX2
//
//go:noescape
func vecScaleF32SIMD(dst, a *float32, scale float32, n int)
