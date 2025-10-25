package kernels

import "math"

// SiLU applies the SiLU (Swish) activation function
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
func SiLU(dst, src []float32, n int) {
	for i := 0; i < n; i++ {
		x := src[i]
		// SiLU(x) = x * sigmoid(x)
		sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		dst[i] = x * sigmoid
	}
}

// GELU applies the GELU activation function (tanh approximation)
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
func GELU(dst, src []float32, n int) {
	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	const coeff = 0.044715

	for i := 0; i < n; i++ {
		x := src[i]
		x3 := x * x * x
		inner := sqrt2OverPi * (x + coeff*x3)
		tanh := float32(math.Tanh(float64(inner)))
		dst[i] = 0.5 * x * (1.0 + tanh)
	}
}

// GELUQuick applies a faster approximation of GELU
// GELU(x) ≈ x * sigmoid(1.702 * x)
func GELUQuick(dst, src []float32, n int) {
	const scale = 1.702

	for i := 0; i < n; i++ {
		x := src[i]
		sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-scale*x))))
		dst[i] = x * sigmoid
	}
}

// ReLU applies the ReLU activation function
// ReLU(x) = max(0, x)
func ReLU(dst, src []float32, n int) {
	for i := 0; i < n; i++ {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

// Softmax applies softmax activation
// softmax(x)_i = exp(x_i) / sum(exp(x_j))
func Softmax(dst, src []float32, n int) {
	if n == 0 {
		return
	}

	// Find max for numerical stability
	maxVal := src[0]
	for i := 1; i < n; i++ {
		if src[i] > maxVal {
			maxVal = src[i]
		}
	}

	// Compute exp and sum
	sum := float32(0)
	for i := 0; i < n; i++ {
		exp := float32(math.Exp(float64(src[i] - maxVal)))
		dst[i] = exp
		sum += exp
	}

	// Normalize
	invSum := 1.0 / sum
	for i := 0; i < n; i++ {
		dst[i] *= invSum
	}
}
