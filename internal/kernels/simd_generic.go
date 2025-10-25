// +build !amd64 noasm

package kernels

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
