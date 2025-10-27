package kernels

import (
	"fmt"
	"math"
	"testing"
)

func TestMatMulF32(t *testing.T) {
	// Test simple 2x3 * 3x2 = 2x2 multiplication
	a := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	b := []float32{
		7, 8,
		9, 10,
		11, 12,
	}
	expected := []float32{
		1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, // [58, 64]
		4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12, // [139, 154]
	}

	dst := make([]float32, 4)
	MatMulF32(dst, a, b, 2, 3, 2)

	for i, v := range expected {
		if math.Abs(float64(dst[i]-v)) > 1e-5 {
			t.Errorf("MatMulF32: dst[%d] = %f, expected %f", i, dst[i], v)
		}
	}
}

func TestVecDotF32(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	expected := float32(1*5 + 2*6 + 3*7 + 4*8) // 70

	result := VecDotF32(a, b, 4)
	if math.Abs(float64(result-expected)) > 1e-5 {
		t.Errorf("VecDotF32 = %f, expected %f", result, expected)
	}
}

func TestRMSNorm(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	dst := make([]float32, 4)

	// RMS = sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
	// normalized = src / RMS
	rms := float32(math.Sqrt((1*1 + 2*2 + 3*3 + 4*4) / 4.0))

	RMSNorm(dst, src, weight, 1e-6)

	for i, v := range src {
		expected := v / rms
		if math.Abs(float64(dst[i]-expected)) > 1e-4 {
			t.Errorf("RMSNorm: dst[%d] = %f, expected %f", i, dst[i], expected)
		}
	}
}

func TestSoftmax(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	dst := make([]float32, 4)

	Softmax(dst, src, 4)

	// Check sum = 1
	sum := float32(0)
	for _, v := range dst {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum = %f, expected 1.0", sum)
	}

	// Check monotonic (larger input -> larger output)
	for i := 0; i < len(dst)-1; i++ {
		if dst[i] >= dst[i+1] {
			t.Errorf("Softmax not monotonic: dst[%d]=%f >= dst[%d]=%f", i, dst[i], i+1, dst[i+1])
		}
	}
}

func TestSiLU(t *testing.T) {
	src := []float32{0, 1, -1, 2}
	dst := make([]float32, 4)

	SiLU(dst, src, 4)

	// Check some known values
	// SiLU(0) = 0
	if math.Abs(float64(dst[0])) > 1e-5 {
		t.Errorf("SiLU(0) = %f, expected 0", dst[0])
	}

	// SiLU(x) should have same sign as x
	for i, x := range src {
		if x > 0 && dst[i] <= 0 {
			t.Errorf("SiLU(%f) = %f should be positive", x, dst[i])
		}
		if x < 0 && dst[i] >= 0 {
			t.Errorf("SiLU(%f) = %f should be negative", x, dst[i])
		}
	}
}

func TestGELU(t *testing.T) {
	src := []float32{0, 1, -1, 2}
	dst := make([]float32, 4)

	GELU(dst, src, 4)

	// Check GELU(0) ≈ 0
	if math.Abs(float64(dst[0])) > 1e-5 {
		t.Errorf("GELU(0) = %f, expected ~0", dst[0])
	}

	// GELU should be monotonic increasing
	for i := 0; i < len(dst)-1; i++ {
		if src[i] < src[i+1] && dst[i] >= dst[i+1] {
			t.Errorf("GELU not monotonic: src[%d]=%f < src[%d]=%f but dst[%d]=%f >= dst[%d]=%f",
				i, src[i], i+1, src[i+1], i, dst[i], i+1, dst[i+1])
		}
	}
}

func BenchmarkMatMulF32(b *testing.B) {
	M, K, N := 128, 256, 128
	a := make([]float32, M*K)
	mat := make([]float32, K*N)
	dst := make([]float32, M*N)

	for i := range a {
		a[i] = float32(i)
	}
	for i := range mat {
		mat[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulF32(dst, a, mat, M, K, N)
	}
}

func BenchmarkRMSNorm(b *testing.B) {
	n := 4096
	src := make([]float32, n)
	weight := make([]float32, n)
	dst := make([]float32, n)

	for i := range src {
		src[i] = float32(i)
		weight[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RMSNorm(dst, src, weight, 1e-6)
	}
}

func BenchmarkAttention(b *testing.B) {
	// Realistic parameters for Gemma embedding model
	batchSize := 1
	seqLen := 32 // Typical short text
	nHeads := 8
	headDim := 64

	qkvSize := batchSize * seqLen * nHeads * headDim
	Q := make([]float32, qkvSize)
	K := make([]float32, qkvSize)
	V := make([]float32, qkvSize)
	output := make([]float32, qkvSize)
	scratch := make([]float32, seqLen*seqLen)

	// Initialize with random-ish values
	for i := range Q {
		Q[i] = float32(i%100) / 100.0
		K[i] = float32((i+1)%100) / 100.0
		V[i] = float32((i+2)%100) / 100.0
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionWithScale(output, Q, K, V, batchSize, seqLen, nHeads, headDim, nil, 0.125, scratch)
		}
	})

	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MultiHeadAttentionWithScaleScalar(output, Q, K, V, batchSize, seqLen, nHeads, headDim, nil, 0.125, scratch)
		}
	})
}

func TestGELUQuickSIMD(t *testing.T) {
	// Test accuracy vs scalar version
	testCases := []struct {
		name   string
		values []float32
	}{
		{"zeros", []float32{0, 0, 0, 0, 0, 0, 0, 0}},
		{"positive", []float32{0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}},
		{"negative", []float32{-0.1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0}},
		{"mixed", []float32{-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}},
		{"large_positive", []float32{5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}},
		{"large_negative", []float32{-5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0}},
		{"small", []float32{0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		{"unaligned", []float32{1.0, 2.0, 3.0, 4.0, 5.0}}, // Not multiple of 8
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.values)
			src := make([]float32, n)
			copy(src, tc.values)

			dstScalar := make([]float32, n)
			dstSIMD := make([]float32, n)

			// Compute with both versions
			GELUQuick(dstScalar, src, n)
			GELUQuickSIMD(dstSIMD, src, n)

			// Compare results
			var maxError float32
			var sumSquaredError float64
			for i := 0; i < n; i++ {
				diff := dstSIMD[i] - dstScalar[i]
				absDiff := float32(math.Abs(float64(diff)))
				if absDiff > maxError {
					maxError = absDiff
				}
				sumSquaredError += float64(diff * diff)
			}
			rmse := math.Sqrt(sumSquaredError / float64(n))

			// Check accuracy: max error should be < 0.01
			if maxError > 0.01 {
				t.Errorf("Max error too high: %f (threshold: 0.01)", maxError)
				for i := 0; i < n; i++ {
					t.Logf("  x=%f: scalar=%f simd=%f diff=%f",
						src[i], dstScalar[i], dstSIMD[i], dstSIMD[i]-dstScalar[i])
				}
			}

			// RMSE should be very small
			if rmse > 0.005 {
				t.Errorf("RMSE too high: %f (threshold: 0.005)", rmse)
			}

			t.Logf("Max error: %.6f, RMSE: %.6f", maxError, rmse)
		})
	}
}

func TestGELUQuickSIMDProperties(t *testing.T) {
	// Test mathematical properties
	t.Run("zero_input", func(t *testing.T) {
		src := []float32{0}
		dst := make([]float32, 1)
		GELUQuickSIMD(dst, src, 1)
		// GELU(0) should be close to 0
		if math.Abs(float64(dst[0])) > 1e-5 {
			t.Errorf("GELU(0) = %f, expected ~0", dst[0])
		}
	})

	t.Run("positive_values", func(t *testing.T) {
		// For large positive x, GELU(x) ≈ x
		src := []float32{5.0, 10.0}
		dst := make([]float32, 2)
		GELUQuickSIMD(dst, src, 2)
		for i, x := range src {
			// Should be close to x
			ratio := dst[i] / x
			if ratio < 0.95 || ratio > 1.0 {
				t.Errorf("GELU(%f) = %f, ratio = %f (expected ~1.0)", x, dst[i], ratio)
			}
		}
	})

	t.Run("negative_values", func(t *testing.T) {
		// For large negative x, GELU(x) ≈ 0
		src := []float32{-5.0, -10.0}
		dst := make([]float32, 2)
		GELUQuickSIMD(dst, src, 2)
		for i, x := range src {
			// Should be close to 0
			if math.Abs(float64(dst[i])) > 0.01 {
				t.Errorf("GELU(%f) = %f, expected ~0", x, dst[i])
			}
		}
	})

	t.Run("monotonic_positive_region", func(t *testing.T) {
		// GELU should be monotonically increasing for x > 0
		// Note: GELU is NOT globally monotonic - it has a local minimum around x=-0.17
		src := []float32{0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0}
		dst := make([]float32, len(src))
		GELUQuickSIMD(dst, src, len(src))
		for i := 0; i < len(dst)-1; i++ {
			if dst[i] >= dst[i+1] {
				t.Errorf("Not monotonic in positive region: GELU(%f)=%f >= GELU(%f)=%f",
					src[i], dst[i], src[i+1], dst[i+1])
			}
		}
	})
}

func BenchmarkGELUQuick(b *testing.B) {
	n := 3072 // Typical FFN intermediate dimension
	src := make([]float32, n)
	dst := make([]float32, n)

	for i := range src {
		src[i] = float32(i%100-50) / 10.0 // Range [-5, 5]
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GELUQuick(dst, src, n)
	}
}

func BenchmarkGELUQuickSIMD(b *testing.B) {
	n := 3072 // Typical FFN intermediate dimension
	src := make([]float32, n)
	dst := make([]float32, n)

	for i := range src {
		src[i] = float32(i%100-50) / 10.0 // Range [-5, 5]
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GELUQuickSIMD(dst, src, n)
	}
}

func TestVecMulF32(t *testing.T) {
	testCases := []struct {
		name string
		n    int
	}{
		{"small_7", 7},   // Test tail loop (< 8)
		{"exact_8", 8},   // Test exactly 8 elements
		{"medium_16", 16}, // Test 2 SIMD iterations
		{"large_100", 100}, // Test mixed SIMD + tail
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float32, tc.n)
			b := make([]float32, tc.n)
			dst := make([]float32, tc.n)
			expected := make([]float32, tc.n)

			// Initialize test data
			for i := 0; i < tc.n; i++ {
				a[i] = float32(i + 1)
				b[i] = float32(i + 2)
				expected[i] = a[i] * b[i]
			}

			VecMulF32(dst, a, b, tc.n)

			// Verify results
			for i := 0; i < tc.n; i++ {
				if math.Abs(float64(dst[i]-expected[i])) > 1e-5 {
					t.Errorf("VecMulF32[%d] = %f, expected %f", i, dst[i], expected[i])
				}
			}
		})
	}
}

func TestVecAddF32(t *testing.T) {
	testCases := []struct {
		name string
		n    int
	}{
		{"small_7", 7},   // Test tail loop (< 8)
		{"exact_8", 8},   // Test exactly 8 elements
		{"medium_16", 16}, // Test 2 SIMD iterations
		{"large_100", 100}, // Test mixed SIMD + tail
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float32, tc.n)
			b := make([]float32, tc.n)
			dst := make([]float32, tc.n)
			expected := make([]float32, tc.n)

			// Initialize test data
			for i := 0; i < tc.n; i++ {
				a[i] = float32(i + 1)
				b[i] = float32(i + 2)
				expected[i] = a[i] + b[i]
			}

			VecAddF32(dst, a, b, tc.n)

			// Verify results
			for i := 0; i < tc.n; i++ {
				if math.Abs(float64(dst[i]-expected[i])) > 1e-5 {
					t.Errorf("VecAddF32[%d] = %f, expected %f", i, dst[i], expected[i])
				}
			}
		})
	}
}

func TestVecScaleF32(t *testing.T) {
	testCases := []struct {
		name  string
		n     int
		scale float32
	}{
		{"small_7", 7, 2.5},   // Test tail loop (< 8)
		{"exact_8", 8, 3.0},   // Test exactly 8 elements
		{"medium_16", 16, 0.5}, // Test 2 SIMD iterations
		{"large_100", 100, 1.5}, // Test mixed SIMD + tail
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float32, tc.n)
			dst := make([]float32, tc.n)
			expected := make([]float32, tc.n)

			// Initialize test data
			for i := 0; i < tc.n; i++ {
				a[i] = float32(i + 1)
				expected[i] = a[i] * tc.scale
			}

			VecScaleF32(dst, a, tc.scale, tc.n)

			// Verify results
			for i := 0; i < tc.n; i++ {
				if math.Abs(float64(dst[i]-expected[i])) > 1e-5 {
					t.Errorf("VecScaleF32[%d] = %f, expected %f", i, dst[i], expected[i])
				}
			}
		})
	}
}

func BenchmarkVecMulF32(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			bVec := make([]float32, size)
			dst := make([]float32, size)

			for i := range a {
				a[i] = float32(i)
				bVec[i] = float32(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VecMulF32(dst, a, bVec, size)
			}
		})
	}
}

func BenchmarkVecAddF32(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			bVec := make([]float32, size)
			dst := make([]float32, size)

			for i := range a {
				a[i] = float32(i)
				bVec[i] = float32(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VecAddF32(dst, a, bVec, size)
			}
		})
	}
}

func BenchmarkVecScaleF32(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			dst := make([]float32, size)
			scale := float32(2.5)

			for i := range a {
				a[i] = float32(i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VecScaleF32(dst, a, scale, size)
			}
		})
	}
}

// Scalar reference implementations for benchmarking
func vecMulF32Scalar(dst, a, b []float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func vecAddF32Scalar(dst, a, b []float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func vecScaleF32Scalar(dst, a []float32, scale float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = a[i] * scale
	}
}

func BenchmarkVecMulF32Scalar(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			bVec := make([]float32, size)
			dst := make([]float32, size)

			for i := range a {
				a[i] = float32(i)
				bVec[i] = float32(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vecMulF32Scalar(dst, a, bVec, size)
			}
		})
	}
}

func BenchmarkVecAddF32Scalar(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			bVec := make([]float32, size)
			dst := make([]float32, size)

			for i := range a {
				a[i] = float32(i)
				bVec[i] = float32(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vecAddF32Scalar(dst, a, bVec, size)
			}
		})
	}
}

func BenchmarkVecScaleF32Scalar(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n%d", size), func(b *testing.B) {
			a := make([]float32, size)
			dst := make([]float32, size)
			scale := float32(2.5)

			for i := range a {
				a[i] = float32(i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vecScaleF32Scalar(dst, a, scale, size)
			}
		})
	}
}
