package kernels

// MultiHeadAttentionWithScaleScalar is the original scalar implementation for benchmarking
// This version uses scalar dot products instead of SIMD for performance comparison
func MultiHeadAttentionWithScaleScalar(
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

					// Scalar dot product
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
