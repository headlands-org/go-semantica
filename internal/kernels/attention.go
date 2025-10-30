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
	// Use 0 threshold to disable parallelism in non-chunked version
	multiHeadAttentionInternal(output, Q, K, V, batchSize, seqLen, nHeads, headDim, mask, scale, scratch, 0, nil, 0)
}

// MultiHeadAttentionChunked computes attention while partitioning the key/value
// sequence into contiguous chunks that can be evaluated in parallel via the
// provided task runner. When chunkSize <= 0, chunkSize >= seqLen, or runTasks is
// nil the function falls back to the serial implementation.
//
// For single-query inference (seqLen=1), heads are parallelized instead of chunks.
// For multi-query, both chunking and head parallelism can be combined.
//
// minHeadsForParallel: Minimum number of heads required to enable head parallelism.
// Set to 8 for multi-query, 16 for single-query for optimal performance.
func MultiHeadAttentionChunked(
	output, Q, K, V []float32,
	batchSize, seqLen, nHeads, headDim int,
	mask []float32,
	scale float32,
	scratch []float32,
	chunkSize int,
	runTasks func(...func()),
	minHeadsForParallel int,
) {
	multiHeadAttentionInternal(output, Q, K, V, batchSize, seqLen, nHeads, headDim, mask, scale, scratch, chunkSize, runTasks, minHeadsForParallel)
}

func multiHeadAttentionInternal(
	output, Q, K, V []float32,
	batchSize, seqLen, nHeads, headDim int,
	mask []float32,
	scale float32,
	scratch []float32,
	chunkSize int,
	runTasks func(...func()),
	minHeadsForParallel int,
) {

	if seqLen == 0 || nHeads == 0 || headDim == 0 || batchSize == 0 {
		return
	}

	headStride := nHeads * headDim
	headScratchStride := seqLen * seqLen

	// Determine parallelization strategy:
	// 1. For single-query (seqLen=1), only parallelize if many heads (threshold × 2)
	//    - Each head has minimal work, so need enough heads to overcome dispatch overhead
	// 2. For multi-query with many heads (>= threshold), parallelize with optional chunking
	// 3. Otherwise use existing chunking strategy
	singleQueryThreshold := minHeadsForParallel * 2 // Double threshold for single-query
	parallelizeHeads := runTasks != nil && ((seqLen == 1 && nHeads >= singleQueryThreshold) || (seqLen > 1 && nHeads >= minHeadsForParallel))
	chunked := runTasks != nil && chunkSize > 0 && chunkSize < seqLen

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchBase := (b * seqLen) * headStride

		if parallelizeHeads {
			// Parallelize across heads (groups of 2-4 heads per task)
			// For single-query: pure head parallelism
			// For multi-query with chunking: heads can be processed in parallel
			headsPerTask := 2
			if nHeads >= 16 {
				headsPerTask = 4
			}
			if seqLen == 1 {
				// Single query: maximize head parallelism
				headsPerTask = 1
			}

			numTasks := (nHeads + headsPerTask - 1) / headsPerTask
			headTasks := make([]func(), numTasks)

			for taskIdx := 0; taskIdx < numTasks; taskIdx++ {
				hStart := taskIdx * headsPerTask
				hEnd := hStart + headsPerTask
				if hEnd > nHeads {
					hEnd = nHeads
				}

				// Capture loop variables
				headStart := hStart
				headEnd := hEnd

				headTasks[taskIdx] = func() {
					processHeadRange(output, Q, K, V, mask, scratch, scale,
						batchBase, headStart, headEnd, seqLen, nHeads, headDim,
						headStride, headScratchStride, chunkSize, chunked, runTasks)
				}
			}

			runTasks(headTasks...)
		} else {
			// Serial head processing with optional chunking
			processHeadRange(output, Q, K, V, mask, scratch, scale,
				batchBase, 0, nHeads, seqLen, nHeads, headDim,
				headStride, headScratchStride, chunkSize, chunked, runTasks)
		}
	}
}

// processHeadRange processes a range of attention heads [headStart, headEnd)
// This function can be called in parallel for different head ranges.
func processHeadRange(
	output, Q, K, V, mask, scratch []float32,
	scale float32,
	batchBase, headStart, headEnd, seqLen, nHeads, headDim int,
	headStride, headScratchStride, chunkSize int,
	chunked bool,
	runTasks func(...func()),
) {
	// Process chunk computations for all heads in this range if chunking is enabled
	if chunked {
		var chunkTasks []func()

		for h := headStart; h < headEnd; h++ {
			headOffset := batchBase + h*headDim
			headScores := scratch[h*headScratchStride : (h+1)*headScratchStride]

			hOffset := headOffset
			scoresView := headScores

			for colStart := 0; colStart < seqLen; colStart += chunkSize {
				start := colStart
				end := start + chunkSize
				if end > seqLen {
					end = seqLen
				}

				s := start
				e := end
				chunkTasks = append(chunkTasks, func() {
					computeAttentionScoresRange(scoresView, Q, K, mask, scale, hOffset, headStride, seqLen, headDim, s, e)
				})
			}
		}

		// Run chunk tasks serially within this head range
		// (they're already part of a parallel head task)
		for _, task := range chunkTasks {
			task()
		}
	}

	// Process each head in the range
	for h := headStart; h < headEnd; h++ {
		headOffset := batchBase + h*headDim
		headScores := scratch[h*headScratchStride : (h+1)*headScratchStride]

		// Compute attention scores if not chunked
		if !chunked {
			computeAttentionScoresRange(headScores, Q, K, mask, scale, headOffset, headStride, seqLen, headDim, 0, seqLen)
		}

		var scaledBuf []float32
		if hasAVX2 && headDim >= 8 {
			scaledBuf = make([]float32, headDim)
		}

		// Apply softmax to each row
		for i := 0; i < seqLen; i++ {
			row := headScores[i*seqLen : i*seqLen+seqLen]
			Softmax(row, row, seqLen)
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
				weight := headScores[i*seqLen+j]
				if weight == 0 {
					vOffset += headStride
					continue
				}

				vRow := V[vOffset : vOffset+headDim]
				axpyAccum(outRow, vRow, weight, scaledBuf)
				vOffset += headStride
			}
		}
	}
}

func computeAttentionScoresRange(
	scores, Q, K, mask []float32,
	scale float32,
	headOffset, headStride, seqLen, headDim int,
	colStart, colEnd int,
) {
	if colStart < 0 {
		colStart = 0
	}
	if colEnd > seqLen {
		colEnd = seqLen
	}
	if colStart >= colEnd {
		return
	}

	for i := 0; i < seqLen; i++ {
		qOffset := headOffset + i*headStride
		qRow := Q[qOffset : qOffset+headDim]
		scoresRow := scores[i*seqLen+colStart : i*seqLen+colEnd]

		kOffset := headOffset + colStart*headStride

		for j := colStart; j < colEnd; j++ {
			kRow := K[kOffset : kOffset+headDim]

			var score float32
			if hasAVX2 && headDim >= 8 {
				score = dotProductSIMD(qRow, kRow, headDim)
			} else {
				score = dotProductInline(qRow, kRow)
			}

			score *= scale

			if mask != nil {
				score += mask[i*seqLen+j]
			}

			scoresRow[j-colStart] = score
			kOffset += headStride
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

// axpyAccum performs out += weight * vec, using SIMD helpers when available.
func axpyAccum(out, vec []float32, weight float32, scaled []float32) {
	if weight == 0 {
		return
	}

	n := len(out)
	if n <= 0 {
		return
	}

	if hasAVX2 && n >= 8 && scaled != nil {
		VecScaleF32(scaled, vec, weight, n)
		VecAddF32(out, out, scaled, n)
		return
	}

	for i := 0; i < n; i++ {
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
	MeanPoolingPartial(dst, src, seqLen, embDim, embDim)
}

// MeanPoolingPartial pools only the first outDim dimensions.
// dst must have length >= outDim.
func MeanPoolingPartial(dst, src []float32, seqLen, embDim, outDim int) {
	if outDim > embDim {
		outDim = embDim
	}
	// Zero output
	for i := 0; i < outDim; i++ {
		dst[i] = 0
	}

	// Sum
	for s := 0; s < seqLen; s++ {
		offset := s * embDim
		for i := 0; i < outDim; i++ {
			dst[i] += src[offset+i]
		}
	}

	// Average
	scale := 1.0 / float32(seqLen)
	for i := 0; i < outDim; i++ {
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
