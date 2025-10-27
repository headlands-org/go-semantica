//go:build arm64 && !noasm

package kernels

import "golang.org/x/sys/cpu"

// SIMD support flags for ARM64
var (
	hasNEON = cpu.ARM64.HasASIMD   // Advanced SIMD (NEON) - always available on ARM64
	hasSDOT = cpu.ARM64.HasASIMDDP // Dot Product instructions (ARMv8.2+)
)

// dotProductNEONAsm is the ARM NEON assembly implementation
// Processes 4 float32 values per iteration using NEON FMLA (fused multiply-add)
//
//go:noescape
func dotProductNEONAsm(a, b *float32, n int) float32

// dotProductSDOTAsm is the ARM NEON SDOT assembly implementation for INT8
// Processes 16 int8 values per iteration using SDOT instruction
// Returns int32 accumulator (caller handles scaling)
//
//go:noescape
func dotProductSDOTAsm(a, b *int8, n int) int32

// matmulInnerLoopAsm processes full 32-element blocks using the fastest available instructions
// on ARM64. Implemented in simd_arm64_matmul.s, falls back to Go when assembly is disabled.
//
//go:noescape
func matmulInnerLoopAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32

// dotProductINT8Asm is the direct assembly call without dispatcher overhead
// IMPORTANT: Caller must ensure slices are valid and n >= 16 for SIMD
func dotProductINT8Asm(a, b *int8, n int) int32 {
	// Direct call to NEON SDOT - no checks, no branches
	return dotProductSDOTAsm(a, b, n)
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

	// Use NEON if available (processes 4 floats at a time)
	// NEON is always available on ARM64, but we check anyway for safety
	if hasNEON && n >= 4 {
		return dotProductNEONAsm(&a[0], &b[0], n)
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
func dotProductINT8SIMD(a, b []int8, n int) int32 {
	if n == 0 {
		return 0
	}

	// Ensure slices are large enough
	if len(a) < n || len(b) < n {
		panic("dotProductINT8SIMD: slice too small")
	}

	// Use SDOT if available (processes 16 int8s at a time, ARM v8.2+)
	// Most Apple Silicon supports this (A12+ / M1+)
	if hasSDOT && n >= 16 {
		return dotProductSDOTAsm(&a[0], &b[0], n)
	}

	// Fallback to scalar INT32 accumulation
	sum := int32(0)
	for i := 0; i < n; i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}
