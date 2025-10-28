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

// GELUQuickSIMD applies a SIMD-optimized version of GELUQuick using polynomial approximation
// GELU(x) ≈ x * sigmoid(1.702 * x)
// Uses fast exp approximation via 2^x identity: exp(x) = 2^(x*log2(e))
// Approximates 2^x using polynomial + bit manipulation
func GELUQuickSIMD(dst, src []float32, n int) {
	const scale = 1.702
	const log2e = 1.442695041 // log2(e)

	// Process in blocks of 8 for better vectorization
	i := 0
	for ; i+8 <= n; i += 8 {
		for j := 0; j < 8; j++ {
			x := src[i+j]
			sx := scale * x

			// Fast sigmoid approximation using 2^x identity
			// sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
			// Use polynomial approximation for 2^x for fractional part

			var sigmoid float32
			if sx >= 0 {
				// Positive path: sigmoid(x) = 1 / (1 + exp(-x))
				negSx := -sx
				exp_negSx := fastExp(negSx)
				sigmoid = 1.0 / (1.0 + exp_negSx)
			} else {
				// Negative path: sigmoid(x) = exp(x) / (1 + exp(x))
				// More numerically stable for negative x
				exp_sx := fastExp(sx)
				sigmoid = exp_sx / (1.0 + exp_sx)
			}

			dst[i+j] = x * sigmoid
		}
	}

	// Handle remaining elements
	for ; i < n; i++ {
		x := src[i]
		sx := scale * x

		var sigmoid float32
		if sx >= 0 {
			exp_negSx := fastExp(-sx)
			sigmoid = 1.0 / (1.0 + exp_negSx)
		} else {
			exp_sx := fastExp(sx)
			sigmoid = exp_sx / (1.0 + exp_sx)
		}

		dst[i] = x * sigmoid
	}
}

// fastExp approximates exp(x) using polynomial and bit manipulation
// Based on the identity: exp(x) = 2^(x * log2(e))
// Accurate to ~0.5% for |x| < 5
func fastExp(x float32) float32 {
	const log2e = 1.442695041

	// Clamp for numerical stability
	if x < -10 {
		return 0
	}
	if x > 10 {
		return 22026.4657948 // exp(10)
	}

	// Convert to 2^y where y = x * log2(e)
	y := x * log2e

	// Split into integer and fractional parts
	intPart := int32(y)
	fracPart := y - float32(intPart)

	// Polynomial approximation for 2^frac (frac in [0,1])
	// 2^x ≈ 1 + x*(0.693147 + x*(0.240227 + x*(0.055504 + x*0.009676)))
	// Coefficients from minimax polynomial fit
	const c1 = 0.693147180559945
	const c2 = 0.240226506959101
	const c3 = 0.055504108664821
	const c4 = 0.009676036358193

	poly := 1.0 + fracPart*(c1+fracPart*(c2+fracPart*(c3+fracPart*c4)))

	// Combine with integer part using bit manipulation
	// 2^intPart is done by shifting the exponent bits
	// float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits
	// Exponent bias is 127

	if intPart >= -126 && intPart <= 127 {
		// Normal range - use bit manipulation
		exponentBits := uint32(intPart+127) << 23
		scale2int := math.Float32frombits(exponentBits)
		return poly * scale2int
	} else if intPart < -126 {
		// Underflow
		return 0
	} else {
		// Overflow
		return 3.402823e+38 // ~max float32
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

	sum := float32(0)
	i := 0
	for ; i+3 < n; i += 4 {
		v0 := src[i] - maxVal
		v1 := src[i+1] - maxVal
		v2 := src[i+2] - maxVal
		v3 := src[i+3] - maxVal

		e0 := fastExp(v0)
		e1 := fastExp(v1)
		e2 := fastExp(v2)
		e3 := fastExp(v3)

		dst[i] = e0
		dst[i+1] = e1
		dst[i+2] = e2
		dst[i+3] = e3

		sum += e0 + e1 + e2 + e3
	}

	for ; i < n; i++ {
		e := fastExp(src[i] - maxVal)
		dst[i] = e
		sum += e
	}

	if sum == 0 {
		inv := 1.0 / float32(n)
		for i := 0; i < n; i++ {
			dst[i] = inv
		}
		return
	}

	invSum := 1.0 / sum
	for i := 0; i < n; i++ {
		dst[i] *= invSum
	}
}
