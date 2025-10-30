// Package runtime provides the execution engine for GGUF models
package runtime

import (
	"fmt"
	"math"
	"runtime"
	"sync"

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
	NumKVHeads     int // For Grouped Query Attention
	HeadDim        int
	IntermDim      int // MLP intermediate dimension
	MaxSeqLen      int
	NormEps        float32
	RoPEBase       float32
	AttentionScale float32 // Gemma-specific: query scaling factor
	PoolingType    string  // "cls", "mean", or "max"
	UseRoPE        bool
	UseBias        bool
}

// ParallelismConfig controls adaptive thresholds for serial vs parallel execution.
// These thresholds determine when parallelization overhead is justified by the workload size.
// Values are tuned empirically through benchmarking on sequences ranging from 1-512 tokens.
type ParallelismConfig struct {
	// MinRoPEWorkForParallel: Parallelize RoPE only if seqLen × nHeads >= threshold.
	// Below this, serial execution is faster due to task dispatch overhead (~200ns/task).
	// Default 64: Optimized for typical embedding models (8-32 heads).
	// - At 64 work units: ~2-4 parallel tasks, break-even point for dispatch overhead
	// - Below 64: Serial is 10-20% faster (overhead dominates)
	// - Above 64: Parallel is 20-40% faster (compute dominates)
	MinRoPEWorkForParallel int

	// MinHeadsForQKNormParallel: Parallelize Q/K normalization if nHeads >= threshold.
	// Q/K norm processes each head independently (headDim elements, typically 64-128).
	// Default 8: Ensures at least 2 parallel tasks (4 heads per task).
	// - Below 8 heads: Serial is faster (1 task = no benefit from parallelism)
	// - At 8+ heads: 2+ tasks justify dispatch cost, ~15-25% speedup
	MinHeadsForQKNormParallel int

	// MinSeqLenForNormParallel: Parallelize RMSNorm across tokens when seqLen reaches
	// this threshold. The value is tuned dynamically based on GOMAXPROCS and model size
	// so that high-core machines start parallelizing earlier (e.g. at 8-12 tokens)
	// while low-core systems keep the higher threshold to avoid dispatch overhead.
	MinSeqLenForNormParallel int

	// MinTileCountForMatmulParallel: Parallelize tiled matmul if tile count >= threshold.
	// Tiled matmul splits output dimensions into cache-friendly chunks.
	// Default 4: Ensures enough tiles to distribute across workers.
	// - Below 4 tiles: Serial is faster (insufficient parallelism)
	// - At 4+ tiles: 15-30% speedup from parallel tile computation
	// - Tile size is adaptive: 64-256 columns depending on batch size
	MinTileCountForMatmulParallel int

	// MinHeadsForAttentionParallel: Parallelize attention computation across heads if nHeads >= threshold.
	// For single-query (seqLen=1), requires more heads due to minimal work per head.
	// For multi-query (seqLen>1), lower threshold is acceptable.
	// Default 8: Balance between dispatch overhead and parallel benefit.
	// - Single query: Uses 16 as effective threshold (more granular parallelism needed)
	// - Multi query with 8+ heads: ~20-35% speedup from head parallelism
	MinHeadsForAttentionParallel int
}

// Model represents a loaded GGUF embedding model
type Model struct {
	config         ModelConfig
	parallelismCfg ParallelismConfig
	reader         *gguf.Reader
	tokenizer      *tokenizer.Tokenizer

	// Embeddings (stored as Q8_0 for zero-copy)
	tokenEmbedQ8 []byte // Raw Q8_0 data from GGUF

	// Layers (we'll load these on demand or cache them)
	// For simplicity in MVP, we'll extract to float32
	layers []Layer // DEPRECATED: not loaded (uses too much memory)

	// INT8 layers (loaded eagerly during model initialization)
	layersINT8 []*LayerINT8

	// Final normalization (Gemma-specific)
	outputNormWeight []float32 // [embDim]

	// RoPE cache for fast position embeddings
	ropeCache *kernels.RoPECache

	workspacePool *sync.Pool
	workers       *workerPool
}

type modelWorkspace struct {
	hidden     []float32
	residual   []float32
	hiddenINT8 kernels.QuantizedTensorINT8

	pooled []float32

	attention attentionWorkspace
	mlp       mlpWorkspace
}

type attentionWorkspace struct {
	q           []float32
	k           []float32
	v           []float32
	kExpanded   []float32
	vExpanded   []float32
	attnOut     []float32
	scratch     []float32
	positions   []int
	attnOutINT8 kernels.QuantizedTensorINT8
}

type mlpWorkspace struct {
	gate     []float32
	up       []float32
	gateINT8 kernels.QuantizedTensorINT8
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

// DefaultParallelismConfig returns empirically-tuned thresholds for serial vs parallel execution.
// Thresholds adapt to the current runtime (GOMAXPROCS) and the model configuration so that the
// single-document path can parallelize earlier on wide machines without hurting low-core latency.
//
// Tuning methodology:
// - Benchmarked on sequences from 1 to 512 tokens
// - Tested on 4-16 core machines (GOMAXPROCS)
// - Measured per-task dispatch overhead: ~200ns
// - Balanced for single-document latency (batch=1) and throughput (batch≥4)
//
// Key insights:
// - Task dispatch overhead dominates for very small workloads (< 50 elements)
// - Parallel benefits appear at 2-4 parallel tasks minimum
// - Cache effects matter: larger chunks (128-256 elements) show better locality
// - Head-level parallelism (nHeads=8-32) is sweet spot for attention operations
func DefaultParallelismConfig(cfg ModelConfig) ParallelismConfig {
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount < 1 {
		workerCount = 1
	}

	// Earlier RMSNorm parallelism on high-core machines and wide embeddings.
	minSeqLen := 24
	switch {
	case workerCount >= 16:
		minSeqLen = 8
	case workerCount >= 8:
		minSeqLen = 8
	case workerCount >= 4:
		minSeqLen = 12
	default:
		minSeqLen = 20
	}
	if cfg.EmbedDim >= 1024 && minSeqLen > 8 {
		minSeqLen = 8
	}

	// RoPE workload threshold: scale down when more workers are available.
	minRoPEWork := 64
	switch {
	case workerCount >= 16:
		minRoPEWork = 24
	case workerCount >= 12:
		minRoPEWork = 32
	case workerCount >= 8:
		minRoPEWork = 40
	case workerCount >= 4:
		minRoPEWork = 48
	}

	// Allow Q/K head norm parallelism on smaller head counts when we have workers available.
	minHeadsForQKNorm := 8
	if cfg.NumHeads <= 8 {
		minHeadsForQKNorm = 4
	} else if workerCount >= 12 {
		minHeadsForQKNorm = 6
	}

	// Attention head parallelism: relax threshold as soon as head count permits.
	minHeadsForAttention := 8
	if cfg.NumHeads <= 8 {
		minHeadsForAttention = 4
	} else if workerCount >= 8 {
		minHeadsForAttention = 6
	}

	// Matmul tiling: require fewer tiles before fanning out when many workers exist.
	minTileCount := 4
	if workerCount >= 12 {
		minTileCount = 3
	}

	return ParallelismConfig{
		MinRoPEWorkForParallel:        minRoPEWork,
		MinHeadsForQKNormParallel:     minHeadsForQKNorm,
		MinSeqLenForNormParallel:      minSeqLen,
		MinTileCountForMatmulParallel: minTileCount,
		MinHeadsForAttentionParallel:  minHeadsForAttention,
	}
}

// LoadModel loads a GGUF model from file
func LoadModel(path string) (*Model, error) {
	reader, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	model, err := loadModel(reader)
	if err != nil {
		reader.Close()
		return nil, err
	}

	return model, nil
}

// LoadModelFromBytes loads a GGUF model directly from an in-memory image.
func LoadModelFromBytes(data []byte) (*Model, error) {
	reader, err := gguf.OpenBytes(data)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	model, err := loadModel(reader)
	if err != nil {
		reader.Close()
		return nil, err
	}

	return model, nil
}

func loadModel(reader *gguf.Reader) (*Model, error) {
	// Parse config from metadata
	config, err := parseConfig(reader)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Load tokenizer
	tok, err := tokenizer.LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	model := &Model{
		config:         config,
		parallelismCfg: DefaultParallelismConfig(config),
		reader:         reader,
		tokenizer:      tok,
		layers:         make([]Layer, config.NumLayers),
	}

	// Load model weights
	if err := model.loadWeights(); err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Initialize RoPE cache if model uses RoPE
	if config.UseRoPE {
		model.ropeCache = kernels.NewRoPECache(config.HeadDim, config.RoPEBase, config.MaxSeqLen)
	}

	// Load INT8 layers eagerly (not lazily) to avoid sync.Once in hot path
	model.layersINT8 = make([]*LayerINT8, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		layer, err := model.loadLayerINT8(i)
		if err != nil {
			return nil, fmt.Errorf("load INT8 layer %d: %w", i, err)
		}
		model.layersINT8[i] = layer
	}

	model.workspacePool = &sync.Pool{
		New: func() interface{} { return &modelWorkspace{} },
	}

	workerCount := runtime.GOMAXPROCS(0)
	if workerCount > 1 {
		model.workers = newWorkerPool(workerCount)
	}

	return model, nil
}

// Close releases model resources
func (m *Model) Close() error {
	if m.workers != nil {
		m.workers.Close()
		m.workers = nil
	}
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

func (m *Model) runTasks(tasks ...func()) {
	if len(tasks) == 0 {
		return
	}
	if m.workers == nil || len(tasks) == 1 {
		for _, task := range tasks {
			task()
		}
		return
	}
	m.workers.Run(tasks...)
}

// runTasksThreshold parallelizes tasks only if count >= minTasks threshold.
// For fine-grained parallelism (e.g., sub-layer operations), the overhead
// of dispatch/sync can exceed serial execution time for small task counts.
// Typical use: minTasks=4 for matmul tiling, minTasks=2 for attention heads.
func (m *Model) runTasksThreshold(tasks []func(), minTasks int) {
	if len(tasks) == 0 {
		return
	}
	// Run serially if below threshold or no worker pool available
	if m.workers == nil || len(tasks) < minTasks {
		for _, task := range tasks {
			if task != nil {
				task()
			}
		}
		return
	}
	m.workers.Run(tasks...)
}

// workerPool implements a fixed-size worker pool for parallel task execution.
// Design rationale:
// - Fixed goroutine pool (GOMAXPROCS workers) to minimize scheduling overhead
// - Buffered job channel (3× worker count) to reduce contention
// - Thread-safe concurrent submission via sync.WaitGroup
// - Per-task dispatch overhead: ~200ns single task, ~100-150ns/task for batches
type workerPool struct {
	jobs chan poolJob
}

type poolJob struct {
	fn func()
	wg *sync.WaitGroup
}

func newWorkerPool(size int) *workerPool {
	if size <= 1 {
		return nil
	}
	// Buffer size = 3× worker count reduces contention on job channel.
	// Empirically, 2× caused blocking under heavy sub-layer parallelism,
	// while 4× showed no improvement over 3×. This balances throughput
	// with memory overhead (~24 bytes/slot on 64-bit).
	p := &workerPool{jobs: make(chan poolJob, size*3)}
	for i := 0; i < size; i++ {
		go func() {
			for job := range p.jobs {
				job.fn()
				job.wg.Done()
			}
		}()
	}
	return p
}

func (p *workerPool) Run(tasks ...func()) {
	var wg sync.WaitGroup
	for _, task := range tasks {
		if task == nil {
			continue
		}
		wg.Add(1)
		p.jobs <- poolJob{fn: task, wg: &wg}
	}
	wg.Wait()
}

func (p *workerPool) Close() {
	close(p.jobs)
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

	// Store Q8_0 embeddings directly (zero-copy)
	if embedTensor.DType.String() != "Q8_0" {
		return fmt.Errorf("expected Q8_0 embeddings, got %s", embedTensor.DType)
	}
	m.tokenEmbedQ8 = embedData

	// Skip loading FP32 layers - use INT8 path exclusively for memory efficiency
	// INT8 layers are loaded lazily in ForwardINT8()

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
// Uses INT8 quantized inference for minimal memory footprint
func (m *Model) Forward(tokenIDs []int) ([]float32, error) {
	return m.ForwardWithDim(tokenIDs, m.config.EmbedDim)
}

// ForwardWithDim generates embeddings truncated to targetDim dimensions.
// targetDim must be >0 and ≤ model EmbedDim.
func (m *Model) ForwardWithDim(tokenIDs []int, targetDim int) ([]float32, error) {
	if targetDim <= 0 || targetDim > m.config.EmbedDim {
		return nil, fmt.Errorf("target dimension %d out of range (1..%d)", targetDim, m.config.EmbedDim)
	}
	// Delegate to INT8 path for zero-copy memory efficiency
	return m.forwardINT8WithDim(tokenIDs, targetDim)

	/* FP32 path disabled to minimize memory (was ~800MB, now ~320MB)
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
	*/
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
