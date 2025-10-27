//go:build (!amd64 && !arm64) || noasm

package kernels

import "unsafe"

// SIMD not available on this platform
var (
	hasAVX2   = false
	hasAVX512 = false
)

// dotProductSIMD falls back to scalar on non-AMD64 platforms
func dotProductSIMD(a, b []float32, n int) float32 {
	return dotProductScalar(a, b, n)
}

// dotProductScalar is the portable scalar implementation
func dotProductScalar(a, b []float32, n int) float32 {
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// dotProductINT8SIMD is the INT8 dot product fallback
// Returns int32 accumulator (caller applies scaling)
func dotProductINT8SIMD(a, b []int8, n int) int32 {
	if n == 0 {
		return 0
	}

	// Fallback to scalar INT32 accumulation
	sum := int32(0)
	for i := 0; i < n; i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}

// vecMulF32SIMD is the scalar fallback for element-wise multiplication
// dst[i] = a[i] * b[i]
func vecMulF32SIMD(dst, a, b *float32, n int) {
	// Convert pointers to slices for bounds checking
	dstSlice := unsafe.Slice(dst, n)
	aSlice := unsafe.Slice(a, n)
	bSlice := unsafe.Slice(b, n)

	for i := 0; i < n; i++ {
		dstSlice[i] = aSlice[i] * bSlice[i]
	}
}

// vecAddF32SIMD is the scalar fallback for element-wise addition
// dst[i] = a[i] + b[i]
func vecAddF32SIMD(dst, a, b *float32, n int) {
	// Convert pointers to slices for bounds checking
	dstSlice := unsafe.Slice(dst, n)
	aSlice := unsafe.Slice(a, n)
	bSlice := unsafe.Slice(b, n)

	for i := 0; i < n; i++ {
		dstSlice[i] = aSlice[i] + bSlice[i]
	}
}

// vecScaleF32SIMD is the scalar fallback for vector scaling
// dst[i] = a[i] * scale
func vecScaleF32SIMD(dst, a *float32, scale float32, n int) {
	// Convert pointers to slices for bounds checking
	dstSlice := unsafe.Slice(dst, n)
	aSlice := unsafe.Slice(a, n)

	for i := 0; i < n; i++ {
		dstSlice[i] = aSlice[i] * scale
	}
}
