// +build !noasm

package kernels

import "golang.org/x/sys/cpu"

// SIMD support flags
var (
	hasAVX2   = cpu.X86.HasAVX2
	hasAVX512 = cpu.X86.HasAVX512F
)

// dotProductAVX2Asm is the assembly implementation
// Processes 8 float32 values per iteration using AVX2
//
//go:noescape
func dotProductAVX2Asm(a, b *float32, n int) float32

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
