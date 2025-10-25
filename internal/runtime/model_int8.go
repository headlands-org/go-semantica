package runtime

import (
	"fmt"
	"math"

	"github.com/lth/pure-go-llamas/internal/gguf"
	"github.com/lth/pure-go-llamas/internal/kernels"
)

// LayerINT8 holds raw Q8_0 weight data for INT8 inference (zero-copy)
type LayerINT8 struct {
	// Attention weights as raw Q8_0 bytes (direct views into GGUF)
	qWeightQ8  []byte // [embDim, embDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	qWeightScales []float32

	kWeightQ8  []byte // [embDim, kvDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	kWeightScales []float32

	vWeightQ8  []byte // [embDim, kvDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	vWeightScales []float32

	oWeightQ8  []byte // [embDim, embDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	oWeightScales []float32

	// MLP weights as raw Q8_0 bytes (direct views into GGUF)
	gateWeightQ8 []byte // [embDim, intermDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	gateWeightScales []float32

	upWeightQ8   []byte // [embDim, intermDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	upWeightScales []float32

	downWeightQ8 []byte // [intermDim, embDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	downWeightScales []float32

	// Keep norm weights as FP32 (they're small and need precision)
	attnNormWeight     []float32
	qNormWeight        []float32
	kNormWeight        []float32
	attnPostNormWeight []float32
	ffnNormWeight      []float32
	ffnPostNormWeight  []float32
}

// loadLayerINT8 loads a layer keeping Q8_0 weights as raw bytes (zero-copy)
func (m *Model) loadLayerINT8(layerIdx int) (*LayerINT8, error) {
	prefix := fmt.Sprintf("blk.%d", layerIdx)
	layer := &LayerINT8{}

	// Load norm weights as FP32
	if err := m.loadTensorF32(prefix+".attn_norm.weight", &layer.attnNormWeight); err != nil {
		return nil, err
	}
	if err := m.loadTensorF32(prefix+".ffn_norm.weight", &layer.ffnNormWeight); err != nil {
		return nil, err
	}

	// Load Q/K norm weights (Gemma-specific)
	m.loadTensorF32(prefix+".attn_q_norm.weight", &layer.qNormWeight) // Ignore error
	m.loadTensorF32(prefix+".attn_k_norm.weight", &layer.kNormWeight)
	m.loadTensorF32(prefix+".post_attention_norm.weight", &layer.attnPostNormWeight)
	m.loadTensorF32(prefix+".post_ffw_norm.weight", &layer.ffnPostNormWeight)

	// Load Q8_0 weights as raw bytes (zero-copy views into GGUF)
	var err error
	layer.qWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_q.weight")
	if err != nil {
		return nil, err
	}
	layer.qWeightScales = extractQ8_0Scales(layer.qWeightQ8)

	layer.kWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_k.weight")
	if err != nil {
		return nil, err
	}
	layer.kWeightScales = extractQ8_0Scales(layer.kWeightQ8)

	layer.vWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_v.weight")
	if err != nil {
		return nil, err
	}
	layer.vWeightScales = extractQ8_0Scales(layer.vWeightQ8)

	layer.oWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_output.weight")
	if err != nil {
		return nil, err
	}
	layer.oWeightScales = extractQ8_0Scales(layer.oWeightQ8)

	layer.gateWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_gate.weight")
	if err != nil {
		return nil, err
	}
	layer.gateWeightScales = extractQ8_0Scales(layer.gateWeightQ8)

	layer.upWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_up.weight")
	if err != nil {
		return nil, err
	}
	layer.upWeightScales = extractQ8_0Scales(layer.upWeightQ8)

	layer.downWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_down.weight")
	if err != nil {
		return nil, err
	}
	layer.downWeightScales = extractQ8_0Scales(layer.downWeightQ8)

	return layer, nil
}

// loadTensorQ8Bytes loads Q8_0 tensor as raw bytes (zero-copy)
func (m *Model) loadTensorQ8Bytes(name string) ([]byte, error) {
	tensor, ok := m.reader.GetTensor(name)
	if !ok {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}

	data, err := m.reader.GetTensorData(name)
	if err != nil {
		return nil, err
	}

	// Verify it's Q8_0
	if tensor.DType.String() != "Q8_0" {
		return nil, fmt.Errorf("expected Q8_0, got %s for %s", tensor.DType, name)
	}

	// Return raw bytes (zero-copy view into GGUF)
	return data, nil
}

// ForwardINT8 performs INT8 quantized inference
// This is experimental and may have accuracy tradeoffs
func (m *Model) ForwardINT8(tokenIDs []int) ([]float32, error) {
	seqLen := len(tokenIDs)
	if seqLen == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if seqLen > m.config.MaxSeqLen {
		return nil, fmt.Errorf("sequence too long: %d > %d", seqLen, m.config.MaxSeqLen)
	}

	embDim := m.config.EmbedDim

	// Load INT8 layers lazily (only once, then cache)
	if m.layersINT8 == nil {
		m.layersINT8 = make([]*LayerINT8, m.config.NumLayers)
		for i := 0; i < m.config.NumLayers; i++ {
			layer, err := m.loadLayerINT8(i)
			if err != nil {
				return nil, fmt.Errorf("load INT8 layer %d: %w", i, err)
			}
			m.layersINT8[i] = layer
		}
	}

	// Allocate activations (start as FP32)
	hidden := make([]float32, seqLen*embDim)
	residual := make([]float32, seqLen*embDim)

	// Embed tokens (on-the-fly Q8_0 dequantization for zero-copy)
	bytesPerRow := ((embDim + 31) / 32) * 34 // Q8_0: (numBlocks * 34 bytes)
	for i, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= m.config.VocabSize {
			return nil, fmt.Errorf("token ID out of range: %d", tokenID)
		}
		// Calculate Q8_0 row offset
		rowOffset := tokenID * bytesPerRow
		rowData := m.tokenEmbedQ8[rowOffset : rowOffset+bytesPerRow]

		// Dequantize directly into hidden buffer
		gguf.DequantizeQ8_0Row(hidden[i*embDim:(i+1)*embDim], rowData, embDim)
	}

	// Gemma-specific: scale input embeddings
	scaleFactor := float32(math.Sqrt(float64(embDim)))
	for i := range hidden {
		hidden[i] *= scaleFactor
	}

	// Run through layers with INT8 inference
	for layerIdx := 0; layerIdx < m.config.NumLayers; layerIdx++ {
		layer := m.layersINT8[layerIdx]

		// Save residual
		copy(residual, hidden)

		// Attention pre-norm (FP32)
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				layer.attnNormWeight,
				m.config.NormEps,
			)
		}

		// Quantize hidden state to INT8
		hiddenINT8 := kernels.QuantizeSymmetricINT8(hidden, seqLen, embDim)

		// Run attention with INT8
		m.runAttentionINT8(hidden, &hiddenINT8, layer, seqLen)

		// Post-attention norm if present
		if len(layer.attnPostNormWeight) > 0 {
			for i := 0; i < seqLen; i++ {
				offset := i * embDim
				kernels.RMSNorm(
					hidden[offset:offset+embDim],
					hidden[offset:offset+embDim],
					layer.attnPostNormWeight,
					m.config.NormEps,
				)
			}
		}

		// Residual connection
		for i := 0; i < seqLen*embDim; i++ {
			hidden[i] += residual[i]
		}

		// Save residual
		copy(residual, hidden)

		// FFN pre-norm
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				layer.ffnNormWeight,
				m.config.NormEps,
			)
		}

		// Quantize for MLP
		hiddenINT8 = kernels.QuantizeSymmetricINT8(hidden, seqLen, embDim)

		// Run MLP with INT8
		m.runMLPINT8(hidden, &hiddenINT8, layer, seqLen)

		// Post-FFN norm if present
		if len(layer.ffnPostNormWeight) > 0 {
			for i := 0; i < seqLen; i++ {
				offset := i * embDim
				kernels.RMSNorm(
					hidden[offset:offset+embDim],
					hidden[offset:offset+embDim],
					layer.ffnPostNormWeight,
					m.config.NormEps,
				)
			}
		}

		// Residual connection
		for i := 0; i < seqLen*embDim; i++ {
			hidden[i] += residual[i]
		}
	}

	// Final output norm
	if len(m.outputNormWeight) > 0 {
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				m.outputNormWeight,
				m.config.NormEps,
			)
		}
	}

	// Pool to single embedding
	embedding := make([]float32, embDim)
	kernels.MeanPooling(embedding, hidden, seqLen, embDim)

	// L2 normalize
	l2Normalize(embedding)

	return embedding, nil
}

// runAttentionINT8 runs attention with INT8 activations
func (m *Model) runAttentionINT8(output []float32, hiddenINT8 *kernels.QuantizedTensorINT8, layer *LayerINT8, seqLen int) {
	embDim := m.config.EmbedDim
	nHeads := m.config.NumHeads
	nKVHeads := m.config.NumKVHeads
	headDim := m.config.HeadDim
	kvDim := nKVHeads * headDim

	// Allocate Q, K, V (FP32 outputs from INT8 matmul)
	q := make([]float32, seqLen*embDim)
	k := make([]float32, seqLen*kvDim)
	v := make([]float32, seqLen*kvDim)

	// Project using INT8 x Q8_0 matmul (raw bytes, zero-copy)
	kernels.MatMulQ8_0INT8(q, layer.qWeightQ8, layer.qWeightScales, hiddenINT8, seqLen, embDim, embDim)
	kernels.MatMulQ8_0INT8(k, layer.kWeightQ8, layer.kWeightScales, hiddenINT8, seqLen, embDim, kvDim)
	kernels.MatMulQ8_0INT8(v, layer.vWeightQ8, layer.vWeightScales, hiddenINT8, seqLen, embDim, kvDim)

	// Q/K normalization (FP32)
	if len(layer.qNormWeight) > 0 {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				offset := s*embDim + h*headDim
				kernels.RMSNorm(q[offset:offset+headDim], q[offset:offset+headDim], layer.qNormWeight, m.config.NormEps)
			}
		}
	}
	if len(layer.kNormWeight) > 0 {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nKVHeads; h++ {
				offset := s*kvDim + h*headDim
				kernels.RMSNorm(k[offset:offset+headDim], k[offset:offset+headDim], layer.kNormWeight, m.config.NormEps)
			}
		}
	}

	// Apply RoPE (FP32)
	if m.config.UseRoPE {
		pos := make([]int, seqLen)
		for i := range pos {
			pos[i] = i
		}
		// Use cached RoPE for fast position embeddings
		kernels.ApplyRoPECached(q, seqLen, nHeads, headDim, pos, m.ropeCache)
		kernels.ApplyRoPECached(k, seqLen, nKVHeads, headDim, pos, m.ropeCache)
	}

	// For GQA, expand K, V
	var kExpanded, vExpanded []float32
	if nKVHeads < nHeads {
		kExpanded = make([]float32, seqLen*embDim)
		vExpanded = make([]float32, seqLen*embDim)
		groupSize := nHeads / nKVHeads

		for s := 0; s < seqLen; s++ {
			for kvHead := 0; kvHead < nKVHeads; kvHead++ {
				kSrc := k[s*kvDim+kvHead*headDim : s*kvDim+(kvHead+1)*headDim]
				vSrc := v[s*kvDim+kvHead*headDim : s*kvDim+(kvHead+1)*headDim]

				for g := 0; g < groupSize; g++ {
					qHead := kvHead*groupSize + g
					copy(kExpanded[s*embDim+qHead*headDim:], kSrc)
					copy(vExpanded[s*embDim+qHead*headDim:], vSrc)
				}
			}
		}
	} else {
		kExpanded = k
		vExpanded = v
	}

	// Run attention (FP32)
	attnOut := make([]float32, seqLen*embDim)
	attnScratch := make([]float32, seqLen*seqLen*nHeads)
	kernels.MultiHeadAttentionWithScale(attnOut, q, kExpanded, vExpanded, 1, seqLen, nHeads, headDim, nil, m.config.AttentionScale, attnScratch)

	// Quantize attention output
	attnOutINT8 := kernels.QuantizeSymmetricINT8(attnOut, seqLen, embDim)

	// Project output with INT8 (raw bytes, zero-copy)
	kernels.MatMulQ8_0INT8(output, layer.oWeightQ8, layer.oWeightScales, &attnOutINT8, seqLen, embDim, embDim)
}

// runMLPINT8 runs MLP with INT8 activations
func (m *Model) runMLPINT8(output []float32, hiddenINT8 *kernels.QuantizedTensorINT8, layer *LayerINT8, seqLen int) {
	embDim := m.config.EmbedDim
	intermDim := m.config.IntermDim

	// Allocate gate and up (FP32)
	gate := make([]float32, seqLen*intermDim)
	up := make([]float32, seqLen*intermDim)

	// Gate and up projections with INT8 (raw bytes, zero-copy)
	kernels.MatMulQ8_0INT8(gate, layer.gateWeightQ8, layer.gateWeightScales, hiddenINT8, seqLen, embDim, intermDim)
	kernels.MatMulQ8_0INT8(up, layer.upWeightQ8, layer.upWeightScales, hiddenINT8, seqLen, embDim, intermDim)

	// Apply GELU to gate (FP32) - using fast approximation
	kernels.GELUQuick(gate, gate, seqLen*intermDim)

	// Element-wise multiply
	kernels.VecMulF32(gate, gate, up, seqLen*intermDim)

	// Quantize for down projection
	gateINT8 := kernels.QuantizeSymmetricINT8(gate, seqLen, intermDim)

	// Down projection with INT8 (raw bytes, zero-copy)
	kernels.MatMulQ8_0INT8(output, layer.downWeightQ8, layer.downWeightScales, &gateINT8, seqLen, intermDim, embDim)
}

// extractQ8_0Scales extracts float32 scales from raw Q8_0 quantized data
// Q8_0 format: 34 bytes per block = [2-byte float16 scale] + [32 int8 values]
// Returns one float32 scale per block
func extractQ8_0Scales(data []byte) []float32 {
	// Handle edge case: empty data
	if len(data) == 0 {
		return nil
	}

	const blockSize = 34
	numBlocks := len(data) / blockSize

	// Handle edge case: partial block (shouldn't happen with valid Q8_0 data)
	if numBlocks == 0 {
		return nil
	}

	scales := make([]float32, numBlocks)

	for i := 0; i < numBlocks; i++ {
		offset := i * blockSize
		// Parse the scale using existing Q8_0 block parser
		block := gguf.ParseQ8_0Block(data[offset : offset+blockSize])
		scales[i] = block.Scale
	}

	return scales
}
