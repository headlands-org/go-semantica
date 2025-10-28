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

	if seqLen == 0 || nHeads == 0 || headDim == 0 || batchSize == 0 {
		return
	}

	headStride := nHeads * headDim

	// For each batch and head
	for b := 0; b < batchSize; b++ {
		batchBase := (b * seqLen) * headStride

		for h := 0; h < nHeads; h++ {
			// Offset within slices that corresponds to the current head.
			headOffset := batchBase + h*headDim

			// Compute attention scores: scores[i,j] = Q[i] · K[j]
			for i := 0; i < seqLen; i++ {
				qOffset := headOffset + i*headStride
				qRow := Q[qOffset : qOffset+headDim]
				scoresRow := scratch[i*seqLen : i*seqLen+seqLen]

				kOffset := headOffset

				for j := 0; j < seqLen; j++ {
					kRow := K[kOffset : kOffset+headDim]

					score := dotProductInline(qRow, kRow) * scale

					// Apply mask if provided
					if mask != nil {
						score += mask[i*seqLen+j]
					}

					scoresRow[j] = score
					kOffset += headStride
				}
			}

			// Apply softmax to each row
			for i := 0; i < seqLen; i++ {
				Softmax(scratch[i*seqLen:], scratch[i*seqLen:], seqLen)
			}

			// Compute weighted sum of values: output[i] = sum_j(scores[i,j] * V[j])
			for i := 0; i < seqLen; i++ {
				outOffset := headOffset + i*headStride
				outRow := output[outOffset : outOffset+headDim]

				for d := range outRow {
					outRow[d] = 0
				}

				vOffset := headOffset

				// Accumulate
				for j := 0; j < seqLen; j++ {
					weight := scratch[i*seqLen+j]
					if weight == 0 {
						vOffset += headStride
						continue
					}

					vRow := V[vOffset : vOffset+headDim]
					axpy(outRow, vRow, weight)
					vOffset += headStride
				}
			}
		}
	}
}

// dotProductInline computes the dot product of two equal-length float32 slices
// with simple unrolling to keep ILP high without introducing additional
// allocations. Head dimensions for embedding models tend to be multiples of
// 8/16 (e.g., 64), making this a good fit.
func dotProductInline(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}

	sum0, sum1, sum2, sum3 := float32(0), float32(0), float32(0), float32(0)
	i := 0

	for ; i+16 <= n; i += 16 {
		sum0 += a[i+0]*b[i+0] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		sum1 += a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		sum2 += a[i+8]*b[i+8] + a[i+9]*b[i+9] + a[i+10]*b[i+10] + a[i+11]*b[i+11]
		sum3 += a[i+12]*b[i+12] + a[i+13]*b[i+13] + a[i+14]*b[i+14] + a[i+15]*b[i+15]
	}

	for ; i+4 <= n; i += 4 {
		sum0 += a[i+0]*b[i+0] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}

	for ; i < n; i++ {
		sum0 += a[i] * b[i]
	}

	return ((sum0 + sum1) + (sum2 + sum3))
}

// axpy performs out += weight * vec with loop unrolling for better throughput.
func axpy(out, vec []float32, weight float32) {
	if weight == 0 {
		return
	}

	n := len(out)
	i := 0

	for ; i+8 <= n; i += 8 {
		out[i+0] += weight * vec[i+0]
		out[i+1] += weight * vec[i+1]
		out[i+2] += weight * vec[i+2]
		out[i+3] += weight * vec[i+3]
		out[i+4] += weight * vec[i+4]
		out[i+5] += weight * vec[i+5]
		out[i+6] += weight * vec[i+6]
		out[i+7] += weight * vec[i+7]
	}

	for ; i < n; i++ {
		out[i] += weight * vec[i]
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
