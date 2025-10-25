package kernels

import "math"

// RMSNorm applies RMS normalization
// out[i] = x[i] / RMS(x) * weight[i]
// RMS(x) = sqrt(mean(x^2) + eps)
func RMSNorm(dst, src, weight []float32, eps float32) {
	n := len(src)
	if len(dst) < n || len(weight) < n {
		panic("RMSNorm: buffer size mismatch")
	}

	// Calculate mean of squares
	sumSq := float32(0)
	for i := 0; i < n; i++ {
		sumSq += src[i] * src[i]
	}
	meanSq := sumSq / float32(n)

	// Calculate RMS
	rms := float32(math.Sqrt(float64(meanSq + eps)))

	// Normalize and scale
	for i := 0; i < n; i++ {
		dst[i] = (src[i] / rms) * weight[i]
	}
}

// LayerNorm applies layer normalization
// out[i] = (x[i] - mean(x)) / sqrt(var(x) + eps) * gamma[i] + beta[i]
func LayerNorm(dst, src, gamma, beta []float32, eps float32) {
	n := len(src)
	if len(dst) < n || len(gamma) < n || len(beta) < n {
		panic("LayerNorm: buffer size mismatch")
	}

	// Calculate mean
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += src[i]
	}
	mean := sum / float32(n)

	// Calculate variance
	sumSq := float32(0)
	for i := 0; i < n; i++ {
		diff := src[i] - mean
		sumSq += diff * diff
	}
	variance := sumSq / float32(n)

	// Normalize and scale
	invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
	for i := 0; i < n; i++ {
		normalized := (src[i] - mean) * invStd
		dst[i] = normalized*gamma[i] + beta[i]
	}
}

// RMSNormNoBias applies RMS normalization without bias (only weight scaling)
func RMSNormNoBias(dst, src, weight []float32, eps float32) {
	RMSNorm(dst, src, weight, eps)
}

// RMSNormGemma applies Gemma-specific RMS normalization
// Gemma uses: out[i] = x[i] * (1 + weight[i]) / RMS(x * (1 + weight))
// First scale by (1 + weight), then normalize
func RMSNormGemma(dst, src, weight []float32, eps float32) {
	n := len(src)
	if len(dst) < n || len(weight) < n {
		panic("RMSNormGemma: buffer size mismatch")
	}

	// First scale by (1 + weight)
	scaled := make([]float32, n)
	for i := 0; i < n; i++ {
		scaled[i] = src[i] * (1.0 + weight[i])
	}

	// Calculate mean of squares of scaled values
	sumSq := float32(0)
	for i := 0; i < n; i++ {
		sumSq += scaled[i] * scaled[i]
	}
	meanSq := sumSq / float32(n)

	// Calculate RMS
	rms := float32(math.Sqrt(float64(meanSq + eps)))

	// Normalize
	for i := 0; i < n; i++ {
		dst[i] = scaled[i] / rms
	}
}
