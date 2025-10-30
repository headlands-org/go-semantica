package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func TestMatmulInnerLoopVNNIMatchesBaseline(t *testing.T) {
	if !hasAVX512VNNI {
		t.Skip("AVX512 VNNI not available on this CPU")
	}

	rng := rand.New(rand.NewSource(1234))

	t.Run("simple", func(t *testing.T) {
		blocks := 1
		inputBytes := make([]int8, 32)
		weightBytes := make([]byte, 34)
		scales := []float32{1.0}

		for i := range inputBytes {
			inputBytes[i] = 1
		}
		for i := range weightBytes[2:] {
			weightBytes[2+i] = 0xFF
		}

		got := matmulInnerLoop(&inputBytes[0], &weightBytes[0], &scales[0], blocks)
		want := matmulInnerLoopAsm(&inputBytes[0], &weightBytes[0], &scales[0], blocks)
		if got != want {
			t.Fatalf("simple case mismatch: vnni=%.6f baseline=%.6f", got, want)
		}
	})

	for blocks := 1; blocks <= 8; blocks++ {
		inputBytes := make([]int8, blocks*32)
		weightBytes := make([]byte, blocks*34)
		scales := make([]float32, blocks)

		for i := range inputBytes {
			inputBytes[i] = int8(rng.Intn(255) - 127)
		}
		for i := range weightBytes {
			weightBytes[i] = byte(rng.Intn(256))
		}
		for i := range scales {
			scales[i] = rng.Float32()*0.5 + 0.5 // Positive scale similar to quantized blocks
		}

		got := matmulInnerLoop(&inputBytes[0], &weightBytes[0], &scales[0], blocks)
		want := matmulInnerLoopAsm(&inputBytes[0], &weightBytes[0], &scales[0], blocks)

		if !almostEqual(got, want, 1e-3) {
			t.Fatalf("blocks=%d: vnni=%.6f baseline=%.6f diff=%.6f input=%v weights=%v scale=%.6f",
				blocks, got, want, math.Abs(float64(got-want)), inputBytes, weightBytes[2:], scales[0])
		}
	}
}

func almostEqual(a, b, tol float32) bool {
	diff := float32(math.Abs(float64(a - b)))
	if diff <= tol {
		return true
	}
	// Allow relative tolerance for larger magnitudes
	maxAbs := float32(math.Max(math.Abs(float64(a)), math.Abs(float64(b))))
	if maxAbs == 0 {
		return diff <= tol
	}
	return diff/maxAbs <= tol
}
