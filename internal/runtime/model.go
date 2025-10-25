// Package runtime provides the execution engine for GGUF models
package runtime

import (
	"fmt"
	"math"

	"github.com/lth/pure-go-llamas/internal/gguf"
	"github.com/lth/pure-go-llamas/internal/kernels"
	"github.com/lth/pure-go-llamas/internal/tokenizer"
)

// ModelConfig holds model hyperparameters
type ModelConfig struct {
	VocabSize      int
	EmbedDim       int
	NumLayers      int
	NumHeads       int
	NumKVHeads     int     // For Grouped Query Attention
	HeadDim        int
	IntermDim      int     // MLP intermediate dimension
	MaxSeqLen      int
	NormEps        float32
	RoPEBase       float32
	AttentionScale float32 // Gemma-specific: query scaling factor
	PoolingType    string  // "cls", "mean", or "max"
	UseRoPE        bool
	UseBias        bool
}

// Model represents a loaded GGUF embedding model
type Model struct {
	config    ModelConfig
	reader    *gguf.Reader
	tokenizer *tokenizer.Tokenizer

	// Embeddings
	tokenEmbed []float32 // [vocabSize, embDim]

	// Layers (we'll load these on demand or cache them)
	// For simplicity in MVP, we'll extract to float32
	layers []Layer

	// Final normalization (Gemma-specific)
	outputNormWeight []float32 // [embDim]

	// Buffers for inference (reusable scratch space)
	scratch []float32
}

// Layer represents a transformer encoder layer
type Layer struct {
	// Attention
	attnNormWeight     []float32 // [embDim] - pre-attention norm
	qWeight            []float32 // [embDim, embDim]
	kWeight            []float32 // [embDim, kvDim]
	vWeight            []float32 // [embDim, kvDim]
	oWeight            []float32 // [embDim, embDim]
	qNormWeight        []float32 // [headDim] - Q normalization (Gemma-specific)
	kNormWeight        []float32 // [headDim] - K normalization (Gemma-specific)
	attnPostNormWeight []float32 // [embDim] - post-attention norm (Gemma-specific)

	// MLP
	ffnNormWeight     []float32 // [embDim] - pre-FFN norm
	gateWeight        []float32 // [embDim, intermDim]
	upWeight          []float32 // [embDim, intermDim]
	downWeight        []float32 // [intermDim, embDim]
	ffnPostNormWeight []float32 // [embDim] - post-FFN norm (Gemma-specific)
}

// LoadModel loads a GGUF model from file
func LoadModel(path string) (*Model, error) {
	reader, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	// Parse config from metadata
	config, err := parseConfig(reader)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Load tokenizer
	tok, err := tokenizer.LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	model := &Model{
		config:    config,
		reader:    reader,
		tokenizer: tok,
		layers:    make([]Layer, config.NumLayers),
	}

	// Load model weights
	if err := model.loadWeights(); err != nil {
		reader.Close()
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Allocate scratch buffer
	// Size estimate: max_seq_len * embed_dim * 10 (conservative)
	scratchSize := config.MaxSeqLen * config.EmbedDim * 10
	model.scratch = make([]float32, scratchSize)

	return model, nil
}

// Close releases model resources
func (m *Model) Close() error {
	if m.reader != nil {
		return m.reader.Close()
	}
	return nil
}

// Config returns the model configuration
func (m *Model) Config() ModelConfig {
	return m.config
}

// Tokenizer returns the tokenizer
func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tokenizer
}

// parseConfig extracts model configuration from GGUF metadata
func parseConfig(r *gguf.Reader) (ModelConfig, error) {
	cfg := ModelConfig{
		NormEps:     1e-6,
		RoPEBase:    10000.0,
		MaxSeqLen:   2048,
		PoolingType: "mean",
		UseRoPE:     true,
	}

	// Determine architecture
	arch := ""
	if val, ok := r.GetMetadata("general.architecture"); ok {
		arch = val.(string)
	}

	// Build prefix for architecture-specific keys
	prefix := ""
	if arch != "" {
		prefix = arch + "."
	}

	// Extract required metadata
	if val, ok := r.GetMetadata(prefix + "embedding_length"); ok {
		cfg.EmbedDim = int(val.(uint32))
	} else {
		return cfg, fmt.Errorf("%sembedding_length not found", prefix)
	}

	if val, ok := r.GetMetadata(prefix + "block_count"); ok {
		cfg.NumLayers = int(val.(uint32))
	} else {
		return cfg, fmt.Errorf("%sblock_count not found", prefix)
	}

	if val, ok := r.GetMetadata(prefix + "attention.head_count"); ok {
		cfg.NumHeads = int(val.(uint32))
	} else {
		return cfg, fmt.Errorf("%sattention.head_count not found", prefix)
	}

	// KV heads for Grouped Query Attention
	if val, ok := r.GetMetadata(prefix + "attention.head_count_kv"); ok {
		cfg.NumKVHeads = int(val.(uint32))
	} else {
		// Default: same as num heads (Multi-Head Attention)
		cfg.NumKVHeads = cfg.NumHeads
	}

	cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads

	// Gemma-specific: attention scale = 1/sqrt(head_dim)
	cfg.AttentionScale = 1.0 / float32(math.Sqrt(float64(cfg.HeadDim)))

	if val, ok := r.GetMetadata(prefix + "feed_forward_length"); ok {
		cfg.IntermDim = int(val.(uint32))
	} else {
		// Default: infer from first layer's ffn tensor if available
		if desc, ok := r.GetTensor("blk.0.ffn_up.weight"); ok {
			// ffn_up is [embDim, intermDim]
			cfg.IntermDim = desc.Shape[1]
		} else {
			// Fallback: use 3x for Gemma (not 4x like most transformers)
			cfg.IntermDim = cfg.EmbedDim * 3
		}
	}

	// Optional metadata
	if val, ok := r.GetMetadata(prefix + "context_length"); ok {
		cfg.MaxSeqLen = int(val.(uint32))
	}

	if val, ok := r.GetMetadata(prefix + "attention.layer_norm_rms_epsilon"); ok {
		if f, ok := val.(float32); ok {
			cfg.NormEps = f
		}
	}

	if val, ok := r.GetMetadata(prefix + "rope.freq_base"); ok {
		if f, ok := val.(float32); ok {
			cfg.RoPEBase = f
		}
	}

	// Get vocab size from tokenizer
	if tokens, ok := r.GetMetadata("tokenizer.ggml.tokens"); ok {
		if tokArr, ok := tokens.([]interface{}); ok {
			cfg.VocabSize = len(tokArr)
		}
	} else {
		cfg.VocabSize = len(r.ListTensors()) // Fallback
	}

	return cfg, nil
}

// loadWeights loads and converts model weights to float32
func (m *Model) loadWeights() error {
	// Load token embeddings
	embedTensor, ok := m.reader.GetTensor("token_embd.weight")
	if !ok {
		// Try alternative names
		embedTensor, ok = m.reader.GetTensor("model.embed_tokens.weight")
		if !ok {
			return fmt.Errorf("token embedding tensor not found")
		}
	}

	embedData, err := m.reader.GetTensorData(embedTensor.Name)
	if err != nil {
		return fmt.Errorf("get token embed data: %w", err)
	}

	// Convert to float32
	m.tokenEmbed, err = m.tensorToFloat32(embedTensor, embedData)
	if err != nil {
		return fmt.Errorf("convert token embed: %w", err)
	}

	// Load layers
	for i := 0; i < m.config.NumLayers; i++ {
		if err := m.loadLayer(i); err != nil {
			return fmt.Errorf("load layer %d: %w", i, err)
		}
	}

	// Load output norm (Gemma-specific)
	if err := m.loadTensorF32("output_norm.weight", &m.outputNormWeight); err != nil {
		// Not all models have this, so don't fail
		_ = err
	}

	return nil
}

// loadLayer loads a single transformer layer
func (m *Model) loadLayer(layerIdx int) error {
	prefix := fmt.Sprintf("blk.%d", layerIdx)

	layer := &m.layers[layerIdx]

	// Attention pre-norm
	if err := m.loadTensorF32(prefix+".attn_norm.weight", &layer.attnNormWeight); err != nil {
		return err
	}

	// Attention weights
	if err := m.loadTensorF32(prefix+".attn_q.weight", &layer.qWeight); err != nil {
		return err
	}
	if err := m.loadTensorF32(prefix+".attn_k.weight", &layer.kWeight); err != nil {
		return err
	}
	if err := m.loadTensorF32(prefix+".attn_v.weight", &layer.vWeight); err != nil {
		return err
	}
	if err := m.loadTensorF32(prefix+".attn_output.weight", &layer.oWeight); err != nil {
		return err
	}

	// Q and K normalization (Gemma-specific)
	if err := m.loadTensorF32(prefix+".attn_q_norm.weight", &layer.qNormWeight); err != nil {
		// Not all models have this
		_ = err
	}
	if err := m.loadTensorF32(prefix+".attn_k_norm.weight", &layer.kNormWeight); err != nil {
		// Not all models have this
		_ = err
	}

	// Attention post-norm (Gemma-specific)
	if err := m.loadTensorF32(prefix+".post_attention_norm.weight", &layer.attnPostNormWeight); err != nil {
		// Not all models have this
		_ = err
	}

	// FFN pre-norm
	if err := m.loadTensorF32(prefix+".ffn_norm.weight", &layer.ffnNormWeight); err != nil {
		return err
	}

	// MLP weights (using Gemma naming)
	if err := m.loadTensorF32(prefix+".ffn_gate.weight", &layer.gateWeight); err != nil {
		return err
	}
	if err := m.loadTensorF32(prefix+".ffn_up.weight", &layer.upWeight); err != nil {
		return err
	}
	if err := m.loadTensorF32(prefix+".ffn_down.weight", &layer.downWeight); err != nil {
		return err
	}

	// FFN post-norm (Gemma-specific - named "post_ffw_norm" in GGUF)
	if err := m.loadTensorF32(prefix+".post_ffw_norm.weight", &layer.ffnPostNormWeight); err != nil {
		// Not all models have this
		_ = err
	}

	return nil
}

// loadTensorF32 loads a tensor and converts to float32
func (m *Model) loadTensorF32(name string, dst *[]float32) error {
	tensor, ok := m.reader.GetTensor(name)
	if !ok {
		return fmt.Errorf("tensor not found: %s", name)
	}

	data, err := m.reader.GetTensorData(name)
	if err != nil {
		return err
	}

	*dst, err = m.tensorToFloat32(tensor, data)
	return err
}

// tensorToFloat32 converts a tensor to float32 based on its dtype
func (m *Model) tensorToFloat32(desc *gguf.TensorDesc, data []byte) ([]float32, error) {
	view := gguf.NewTensorView(desc, data)

	switch desc.DType {
	case gguf.DTypeF32:
		return view.AsFloat32()

	case gguf.DTypeQ8_0:
		numElems := view.NumElements()
		return gguf.DequantizeQ8_0(data, numElems), nil

	default:
		return nil, fmt.Errorf("unsupported dtype for conversion: %s", desc.DType)
	}
}

// Forward performs a forward pass to generate embeddings
func (m *Model) Forward(tokenIDs []int) ([]float32, error) {
	seqLen := len(tokenIDs)
	if seqLen == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if seqLen > m.config.MaxSeqLen {
		return nil, fmt.Errorf("sequence too long: %d > %d", seqLen, m.config.MaxSeqLen)
	}

	embDim := m.config.EmbedDim

	// Allocate activations
	// We'll use scratch buffer for intermediate computations
	hidden := make([]float32, seqLen*embDim)
	residual := make([]float32, seqLen*embDim)

	// Embed tokens
	for i, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= m.config.VocabSize {
			return nil, fmt.Errorf("token ID out of range: %d", tokenID)
		}
		offset := tokenID * embDim
		copy(hidden[i*embDim:(i+1)*embDim], m.tokenEmbed[offset:offset+embDim])
	}

	// Gemma-specific: scale input embeddings by sqrt(embDim)
	scaleFactor := float32(math.Sqrt(float64(embDim)))
	for i := range hidden {
		hidden[i] *= scaleFactor
	}

	// Run through transformer layers
	for layerIdx := 0; layerIdx < m.config.NumLayers; layerIdx++ {
		layer := &m.layers[layerIdx]

		// Save residual
		copy(residual, hidden)

		// Attention pre-norm (Gemma-specific)
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				layer.attnNormWeight,
				m.config.NormEps,
			)
		}

		// Self-attention (with Q/K normalization and scaling)
		m.runAttention(hidden, layer, seqLen)

		// Gemma-specific: Attention post-norm
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

		// FFN pre-norm (Gemma-specific)
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				layer.ffnNormWeight,
				m.config.NormEps,
			)
		}

		// MLP (GeGLU variant for Gemma)
		m.runMLP(hidden, layer, seqLen)

		// Gemma-specific: FFN post-norm
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

	// Gemma-specific: Final output norm before pooling
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
	switch m.config.PoolingType {
	case "cls":
		kernels.CLSPooling(embedding, hidden, seqLen, embDim)
	case "mean":
		kernels.MeanPooling(embedding, hidden, seqLen, embDim)
	case "max":
		kernels.MaxPooling(embedding, hidden, seqLen, embDim)
	default:
		return nil, fmt.Errorf("unknown pooling type: %s", m.config.PoolingType)
	}

	// L2 normalize (standard for embedding models)
	l2Normalize(embedding)

	return embedding, nil
}

// l2Normalize normalizes a vector to unit length
func l2Normalize(v []float32) {
	var sumSq float32
	for _, x := range v {
		sumSq += x * x
	}
	if sumSq == 0 {
		return
	}
	norm := float32(math.Sqrt(float64(sumSq)))
	for i := range v {
		v[i] /= norm
	}
}

// runAttention runs the attention mechanism
func (m *Model) runAttention(hidden []float32, layer *Layer, seqLen int) {
	embDim := m.config.EmbedDim
	nHeads := m.config.NumHeads
	nKVHeads := m.config.NumKVHeads
	headDim := m.config.HeadDim
	kvDim := nKVHeads * headDim

	// Allocate Q, K, V from scratch
	qSize := seqLen * embDim
	kvSize := seqLen * kvDim
	q := m.scratch[0:qSize]
	k := m.scratch[qSize : qSize+kvSize]
	v := m.scratch[qSize+kvSize : qSize+2*kvSize]
	attnOut := m.scratch[qSize+2*kvSize : qSize+2*kvSize+qSize]
	attnScratch := m.scratch[qSize+2*kvSize+qSize:]

	// Project Q, K, V (using ggml semantics: weight first, input second)
	kernels.MatMulGGML(q, layer.qWeight, hidden, seqLen, embDim, embDim)
	kernels.MatMulGGML(k, layer.kWeight, hidden, seqLen, embDim, kvDim)
	kernels.MatMulGGML(v, layer.vWeight, hidden, seqLen, embDim, kvDim)

	// Gemma-specific: Normalize Q and K after projection
	if len(layer.qNormWeight) > 0 {
		// Q is [seqLen, nHeads, headDim], normalize per head
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				offset := s*embDim + h*headDim
				kernels.RMSNorm(
					q[offset:offset+headDim],
					q[offset:offset+headDim],
					layer.qNormWeight,
					m.config.NormEps,
				)
			}
		}
	}

	if len(layer.kNormWeight) > 0 {
		// K is [seqLen, nKVHeads, headDim], normalize per head
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nKVHeads; h++ {
				offset := s*kvDim + h*headDim
				kernels.RMSNorm(
					k[offset:offset+headDim],
					k[offset:offset+headDim],
					layer.kNormWeight,
					m.config.NormEps,
				)
			}
		}
	}

	// Apply RoPE if configured
	if m.config.UseRoPE {
		pos := make([]int, seqLen)
		for i := range pos {
			pos[i] = i
		}
		kernels.ApplyRoPE(q, seqLen, nHeads, headDim, pos, m.config.RoPEBase)
		kernels.ApplyRoPE(k, seqLen, nKVHeads, headDim, pos, m.config.RoPEBase)
	}

	// For GQA, we need to expand K, V to match Q heads
	// If nKVHeads == nHeads, this is regular MHA
	// If nKVHeads < nHeads, we replicate KV groups
	if nKVHeads < nHeads {
		// Expand K, V by replicating each KV head for multiple Q heads
		kExpanded := make([]float32, qSize)
		vExpanded := make([]float32, qSize)
		groupSize := nHeads / nKVHeads

		for s := 0; s < seqLen; s++ {
			for kvHead := 0; kvHead < nKVHeads; kvHead++ {
				kSrc := k[s*kvDim+kvHead*headDim : s*kvDim+(kvHead+1)*headDim]
				vSrc := v[s*kvDim+kvHead*headDim : s*kvDim+(kvHead+1)*headDim]

				for g := 0; g < groupSize; g++ {
					qHead := kvHead*groupSize + g
					kDst := kExpanded[s*embDim+qHead*headDim : s*embDim+(qHead+1)*headDim]
					vDst := vExpanded[s*embDim+qHead*headDim : s*embDim+(qHead+1)*headDim]
					copy(kDst, kSrc)
					copy(vDst, vSrc)
				}
			}
		}

		// Run attention with expanded K, V - use attention_scale instead of 1/sqrt(headDim)
		kernels.MultiHeadAttentionWithScale(attnOut, q, kExpanded, vExpanded, 1, seqLen, nHeads, headDim, nil, m.config.AttentionScale, attnScratch)
	} else {
		// Regular MHA - use attention_scale instead of 1/sqrt(headDim)
		kernels.MultiHeadAttentionWithScale(attnOut, q, k, v, 1, seqLen, nHeads, headDim, nil, m.config.AttentionScale, attnScratch)
	}

	// Project output (using ggml semantics: weight first, input second)
	kernels.MatMulGGML(hidden, layer.oWeight, attnOut, seqLen, embDim, embDim)
}

// runMLP runs the MLP/FFN block (GeGLU for Gemma)
func (m *Model) runMLP(hidden []float32, layer *Layer, seqLen int) {
	embDim := m.config.EmbedDim
	intermDim := m.config.IntermDim

	// Allocate from scratch
	gateSize := seqLen * intermDim
	gate := m.scratch[0:gateSize]
	up := m.scratch[gateSize : 2*gateSize]

	// Gate and Up projections (using ggml semantics: weight first, input second)
	kernels.MatMulGGML(gate, layer.gateWeight, hidden, seqLen, embDim, intermDim)
	kernels.MatMulGGML(up, layer.upWeight, hidden, seqLen, embDim, intermDim)

	// Apply GELU to gate
	kernels.GELU(gate, gate, seqLen*intermDim)

	// Element-wise multiply: gate * up
	kernels.VecMulF32(gate, gate, up, seqLen*intermDim)

	// Down projection (using ggml semantics: weight first, input second)
	kernels.MatMulGGML(hidden, layer.downWeight, gate, seqLen, intermDim, embDim)
}
