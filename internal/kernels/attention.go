package kernels

import (
	"math"
)

// ApplyRoPE applies Rotary Position Embedding (RoPE) to query/key tensors
// qk: [seqLen, nHeads, headDim] - interleaved real/imag pairs
// pos: position indices for each token
// dim: head dimension
// base: RoPE base frequency (typically 10000.0)
func ApplyRoPE(qk []float32, seqLen, nHeads, headDim int, pos []int, base float32) {
	if len(pos) != seqLen {
		panic("ApplyRoPE: pos length must equal seqLen")
	}

	halfDim := headDim / 2

	for s := 0; s < seqLen; s++ {
		p := float32(pos[s])

		for h := 0; h < nHeads; h++ {
			offset := (s*nHeads + h) * headDim

			// Apply rotation to each pair of dimensions
			for i := 0; i < halfDim; i++ {
				// Compute frequency
				freq := float32(1.0 / math.Pow(float64(base), float64(2*i)/float64(headDim)))
				theta := p * freq

				cos := float32(math.Cos(float64(theta)))
				sin := float32(math.Sin(float64(theta)))

				// Rotate the pair (i, i+halfDim)
				idx0 := offset + i
				idx1 := offset + i + halfDim

				v0 := qk[idx0]
				v1 := qk[idx1]

				qk[idx0] = v0*cos - v1*sin
				qk[idx1] = v0*sin + v1*cos
			}
		}
	}
}

// MultiHeadAttention computes multi-head self-attention with default scaling
// Q, K, V: [batchSize, seqLen, nHeads, headDim]
// output: [batchSize, seqLen, nHeads, headDim]
// mask: optional attention mask [seqLen, seqLen]
func MultiHeadAttention(
	output, Q, K, V []float32,
	batchSize, seqLen, nHeads, headDim int,
	mask []float32,
	scratch []float32,
) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	MultiHeadAttentionWithScale(output, Q, K, V, batchSize, seqLen, nHeads, headDim, mask, scale, scratch)
}

// MultiHeadAttentionWithScale computes multi-head self-attention with custom scale
// Q, K, V: [batchSize, seqLen, nHeads, headDim]
// output: [batchSize, seqLen, nHeads, headDim]
// mask: optional attention mask [seqLen, seqLen]
// scale: custom attention scale (use 1.0 if Q is already scaled)
func MultiHeadAttentionWithScale(
	output, Q, K, V []float32,
	batchSize, seqLen, nHeads, headDim int,
	mask []float32,
	scale float32,
	scratch []float32,
) {

	// For each batch and head
	for b := 0; b < batchSize; b++ {
		for h := 0; h < nHeads; h++ {
			// Compute attention scores: scores[i,j] = Q[i] · K[j]
			// scores: [seqLen, seqLen]
			for i := 0; i < seqLen; i++ {
				qOffset := ((b*seqLen+i)*nHeads + h) * headDim

				for j := 0; j < seqLen; j++ {
					kOffset := ((b*seqLen+j)*nHeads + h) * headDim

					// Dot product
					score := float32(0)
					for d := 0; d < headDim; d++ {
						score += Q[qOffset+d] * K[kOffset+d]
					}
					score *= scale

					// Apply mask if provided
					if mask != nil {
						score += mask[i*seqLen+j]
					}

					scratch[i*seqLen+j] = score
				}
			}

			// Apply softmax to each row
			for i := 0; i < seqLen; i++ {
				Softmax(scratch[i*seqLen:], scratch[i*seqLen:], seqLen)
			}

			// Compute weighted sum of values: output[i] = sum_j(scores[i,j] * V[j])
			for i := 0; i < seqLen; i++ {
				outOffset := ((b*seqLen+i)*nHeads + h) * headDim

				// Zero output
				for d := 0; d < headDim; d++ {
					output[outOffset+d] = 0
				}

				// Accumulate
				for j := 0; j < seqLen; j++ {
					weight := scratch[i*seqLen+j]
					vOffset := ((b*seqLen+j)*nHeads + h) * headDim

					for d := 0; d < headDim; d++ {
						output[outOffset+d] += weight * V[vOffset+d]
					}
				}
			}
		}
	}
}

// SelfAttentionEncoder computes encoder-style self-attention (no causal mask)
// input: [seqLen, embDim]
// output: [seqLen, embDim]
// Wq, Wk, Wv, Wo: weight matrices
func SelfAttentionEncoder(
	output, input []float32,
	seqLen, embDim, nHeads, headDim int,
	Wq, Wk, Wv, Wo []float32,
	scratch []float32,
) {
	// Project to Q, K, V
	qkvDim := nHeads * headDim

	qOffset := 0
	kOffset := qkvDim * seqLen
	vOffset := 2 * qkvDim * seqLen

	// Q = input @ Wq
	MatMulF32(scratch[qOffset:], input, Wq, seqLen, embDim, qkvDim)

	// K = input @ Wk
	MatMulF32(scratch[kOffset:], input, Wk, seqLen, embDim, qkvDim)

	// V = input @ Wv
	MatMulF32(scratch[vOffset:], input, Wv, seqLen, embDim, qkvDim)

	// Compute attention
	attnOut := scratch[3*qkvDim*seqLen:]
	attnScratch := attnOut[qkvDim*seqLen:]

	MultiHeadAttention(
		attnOut,
		scratch[qOffset:], // Q
		scratch[kOffset:], // K
		scratch[vOffset:], // V
		1, seqLen, nHeads, headDim,
		nil, // no mask for encoder
		attnScratch,
	)

	// Project back: output = attnOut @ Wo
	MatMulF32(output, attnOut, Wo, seqLen, qkvDim, embDim)
}

// Pooling operations

// CLSPooling extracts the [CLS] token embedding (first token)
func CLSPooling(dst, src []float32, seqLen, embDim int) {
	copy(dst[:embDim], src[:embDim])
}

// MeanPooling computes mean pooling over sequence
func MeanPooling(dst, src []float32, seqLen, embDim int) {
	// Zero output
	for i := 0; i < embDim; i++ {
		dst[i] = 0
	}

	// Sum
	for s := 0; s < seqLen; s++ {
		offset := s * embDim
		for i := 0; i < embDim; i++ {
			dst[i] += src[offset+i]
		}
	}

	// Average
	scale := 1.0 / float32(seqLen)
	for i := 0; i < embDim; i++ {
		dst[i] *= scale
	}
}

// MaxPooling computes max pooling over sequence
func MaxPooling(dst, src []float32, seqLen, embDim int) {
	// Initialize with first token
	copy(dst[:embDim], src[:embDim])

	// Take max
	for s := 1; s < seqLen; s++ {
		offset := s * embDim
		for i := 0; i < embDim; i++ {
			if src[offset+i] > dst[i] {
				dst[i] = src[offset+i]
			}
		}
	}
}
