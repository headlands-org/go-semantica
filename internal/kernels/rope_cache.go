package kernels

import "math"

// RoPECache stores pre-computed RoPE frequencies and trigonometric values
type RoPECache struct {
	freqs    []float32 // Pre-computed frequencies [headDim/2]
	cosCache []float32 // Cached cos values [maxPos][headDim/2]
	sinCache []float32 // Cached sin values [maxPos][headDim/2]
	headDim  int
	maxPos   int
}

// NewRoPECache creates and initializes a RoPE cache
func NewRoPECache(headDim int, base float32, maxPos int) *RoPECache {
	halfDim := headDim / 2

	cache := &RoPECache{
		freqs:    make([]float32, halfDim),
		cosCache: make([]float32, maxPos*halfDim),
		sinCache: make([]float32, maxPos*halfDim),
		headDim:  headDim,
		maxPos:   maxPos,
	}

	// Pre-compute frequencies (independent of position)
	for i := 0; i < halfDim; i++ {
		cache.freqs[i] = float32(1.0 / math.Pow(
			float64(base),
			float64(2*i)/float64(headDim)))
	}

	// Pre-compute cos/sin for all positions
	for pos := 0; pos < maxPos; pos++ {
		p := float32(pos)
		for i := 0; i < halfDim; i++ {
			theta := p * cache.freqs[i]
			cache.cosCache[pos*halfDim+i] = float32(math.Cos(float64(theta)))
			cache.sinCache[pos*halfDim+i] = float32(math.Sin(float64(theta)))
		}
	}

	return cache
}

// ApplyRoPECached applies RoPE using pre-computed cos/sin values
func ApplyRoPECached(qk []float32, seqLen, nHeads, headDim int, pos []int, cache *RoPECache) {
	if len(pos) != seqLen {
		panic("ApplyRoPECached: pos length must equal seqLen")
	}

	if cache == nil || cache.headDim != headDim {
		panic("ApplyRoPECached: invalid cache")
	}

	halfDim := headDim / 2

	for s := 0; s < seqLen; s++ {
		p := pos[s]

		// Check if position is in cache range
		if p >= cache.maxPos {
			// Fallback to original computation for positions beyond cache
			// This should be rare for typical inference
			pf := float32(p)
			for h := 0; h < nHeads; h++ {
				offset := (s*nHeads + h) * headDim
				for i := 0; i < halfDim; i++ {
					freq := cache.freqs[i]
					theta := pf * freq
					cos := float32(math.Cos(float64(theta)))
					sin := float32(math.Sin(float64(theta)))

					idx0 := offset + i
					idx1 := offset + i + halfDim
					v0 := qk[idx0]
					v1 := qk[idx1]
					qk[idx0] = v0*cos - v1*sin
					qk[idx1] = v0*sin + v1*cos
				}
			}
			continue
		}

		// Use cached values
		cosPtr := cache.cosCache[p*halfDim:]
		sinPtr := cache.sinCache[p*halfDim:]

		for h := 0; h < nHeads; h++ {
			offset := (s*nHeads + h) * headDim

			for i := 0; i < halfDim; i++ {
				cos := cosPtr[i]
				sin := sinPtr[i]

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
