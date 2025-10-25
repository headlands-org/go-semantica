package kernels

import (
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
