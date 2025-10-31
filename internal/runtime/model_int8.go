package runtime

import (
	"fmt"
	"math"
	goruntime "runtime"

	"github.com/headlands-org/go-semantica/internal/gguf"
	"github.com/headlands-org/go-semantica/internal/kernels"
)

// LayerINT8 holds raw Q8_0 weight data for INT8 inference (zero-copy)
type LayerINT8 struct {
	// Attention weights as raw Q8_0 bytes (direct views into GGUF)
	qWeightQ8 []byte // [embDim, embDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	qWeightScales []float32

	kWeightQ8 []byte // [embDim, kvDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	kWeightScales []float32

	vWeightQ8 []byte // [embDim, kvDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	vWeightScales []float32

	oWeightQ8 []byte // [embDim, embDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	oWeightScales []float32

	// MLP weights as raw Q8_0 bytes (direct views into GGUF)
	gateWeightQ8 []byte // [embDim, intermDim]
	// Pre-converted float32 scales (one per 32-element block, extracted from Q8_0)
	gateWeightScales []float32

	upWeightQ8 []byte // [embDim, intermDim]
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
	embDim := m.config.EmbedDim
	intermDim := m.config.IntermDim
	var err error
	layer.qWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_q.weight")
	if err != nil {
		return nil, err
	}
	layer.qWeightScales = extractQ8_0Scales(layer.qWeightQ8, embDim)

	layer.kWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_k.weight")
	if err != nil {
		return nil, err
	}
	layer.kWeightScales = extractQ8_0Scales(layer.kWeightQ8, embDim)

	layer.vWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_v.weight")
	if err != nil {
		return nil, err
	}
	layer.vWeightScales = extractQ8_0Scales(layer.vWeightQ8, embDim)

	layer.oWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".attn_output.weight")
	if err != nil {
		return nil, err
	}
	layer.oWeightScales = extractQ8_0Scales(layer.oWeightQ8, embDim)

	layer.gateWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_gate.weight")
	if err != nil {
		return nil, err
	}
	layer.gateWeightScales = extractQ8_0Scales(layer.gateWeightQ8, embDim)

	layer.upWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_up.weight")
	if err != nil {
		return nil, err
	}
	layer.upWeightScales = extractQ8_0Scales(layer.upWeightQ8, embDim)

	layer.downWeightQ8, err = m.loadTensorQ8Bytes(prefix + ".ffn_down.weight")
	if err != nil {
		return nil, err
	}
	layer.downWeightScales = extractQ8_0Scales(layer.downWeightQ8, intermDim)

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

// ForwardINT8 performs INT8 quantized inference returning the full embedding dimension.
func (m *Model) ForwardINT8(tokenIDs []int) ([]float32, error) {
	return m.forwardINT8WithDim(tokenIDs, m.config.EmbedDim)
}

// forwardINT8WithDim performs INT8 quantized inference with output truncated to targetDim.
func (m *Model) forwardINT8WithDim(tokenIDs []int, targetDim int) ([]float32, error) {
	seqLen := len(tokenIDs)
	if seqLen == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if seqLen > m.config.MaxSeqLen {
		return nil, fmt.Errorf("sequence too long: %d > %d", seqLen, m.config.MaxSeqLen)
	}

	embDim := m.config.EmbedDim

	ws := m.workspacePool.Get().(*modelWorkspace)
	defer m.workspacePool.Put(ws)

	// Reuse persistent work buffers to avoid per-call allocation churn
	hidden := ensureFloat32(&ws.hidden, seqLen*embDim)
	residual := ensureFloat32(&ws.residual, seqLen*embDim)

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
		m.applyRMSNormParallel(hidden, layer.attnNormWeight, seqLen, embDim)

		// Quantize hidden state to INT8
		kernels.QuantizeSymmetricINT8Into(&ws.hiddenINT8, hidden, seqLen, embDim)

		// Run attention with INT8
		m.runAttentionINT8(ws, hidden, &ws.hiddenINT8, layer, seqLen)

		// Post-attention norm if present
		if len(layer.attnPostNormWeight) > 0 {
			m.applyRMSNormParallel(hidden, layer.attnPostNormWeight, seqLen, embDim)
		}

		// Residual connection
		for i := 0; i < seqLen*embDim; i++ {
			hidden[i] += residual[i]
		}

		// Save residual
		copy(residual, hidden)

		// FFN pre-norm
		m.applyRMSNormParallel(hidden, layer.ffnNormWeight, seqLen, embDim)

		// Quantize for MLP
		kernels.QuantizeSymmetricINT8Into(&ws.hiddenINT8, hidden, seqLen, embDim)

		// Run MLP with INT8
		m.runMLPINT8(ws, hidden, &ws.hiddenINT8, layer, seqLen)

		// Post-FFN norm if present
		if len(layer.ffnPostNormWeight) > 0 {
			m.applyRMSNormParallel(hidden, layer.ffnPostNormWeight, seqLen, embDim)
		}

		// Residual connection
		for i := 0; i < seqLen*embDim; i++ {
			hidden[i] += residual[i]
		}
	}

	// Final output norm
	if len(m.outputNormWeight) > 0 {
		m.applyRMSNormParallel(hidden, m.outputNormWeight, seqLen, embDim)
	}

	// Pool to single embedding
	outDim := targetDim
	if outDim <= 0 || outDim > embDim {
		outDim = embDim
	}

	pooled := ensureFloat32(&ws.pooled, outDim)
	kernels.MeanPoolingPartial(pooled, hidden, seqLen, embDim, outDim)

	// L2 normalize over requested slice
	l2Normalize(pooled[:outDim])

	result := make([]float32, outDim)
	copy(result, pooled[:outDim])

	return result, nil
}

func (m *Model) matMulQ8_0INT8Tiled(dst []float32, weightData []byte, scales []float32, input *kernels.QuantizedTensorINT8, batch, inDim, outDim, tileOut int) {
	// Adaptive tiling strategy: optimize for single-document latency vs batch throughput
	// For single documents (batch=1), use aggressive parallelism with larger tiles
	// For batches (batch≥4), use conservative cache-optimized tiling
	if tileOut <= 0 {
		tileOut = outDim
	}

	// DEBUG: Log parallelization decision
	// fmt.Printf("DEBUG matMulQ8_0INT8Tiled: batch=%d, inDim=%d, outDim=%d, tileOut=%d, workers=%v\n", batch, inDim, outDim, tileOut, m.workers != nil)

	// Fallback to serial execution if no workers or problem is too small
	if m.workers == nil || outDim <= tileOut {
		// fmt.Printf("DEBUG: Using serial execution (workers=%v, outDim=%d, tileOut=%d)\n", m.workers != nil, outDim, tileOut)
		kernels.MatMulQ8_0INT8(dst, weightData, scales, input, batch, inDim, outDim)
		return
	}

	// Adaptive tile size selection based on batch size and problem size
	var chunk int
	workerCount := goruntime.GOMAXPROCS(0)

	// For large batches, use the hint directly (cache-optimized)
	// For single/small batches, use aggressive parallelism
	if batch >= 8 {
		// Large batch: use cache-optimized tiling (typically 64-128)
		chunk = tileOut
		if chunk > outDim {
			chunk = outDim
		}
	} else if batch >= 4 {
		// Medium batch: slightly more aggressive than hint
		chunk = tileOut * 2
		if chunk > outDim/2 {
			chunk = outDim / 2
		}
		if chunk < 64 {
			chunk = 64
		}
	} else {
		// Single/small batch: maximize parallelism for low latency
		chunk = max(64, outDim/workerCount)
		// Safety bound: prevent tiles that are too large (poor cache behavior)
		if chunk > 256 {
			chunk = 256
		}
	}

	// Safety bound: prevent tiles that are too small (overhead dominates)
	if chunk < 32 {
		chunk = 32
	}

	// If computed chunk is too large, fall back to serial execution
	if chunk >= outDim {
		kernels.MatMulQ8_0INT8(dst, weightData, scales, input, batch, inDim, outDim)
		return
	}

	// Zero-initialize output buffer for accumulation
	total := batch * outDim
	for i := range dst[:total] {
		dst[i] = 0
	}

	// Partition work into tiles and execute in parallel
	taskCount := (outDim + chunk - 1) / chunk
	tasks := make([]func(), 0, taskCount)
	for colStart := 0; colStart < outDim; colStart += chunk {
		start := colStart
		end := start + chunk
		if end > outDim {
			end = outDim
		}

		tasks = append(tasks, func() {
			kernels.MatMulQ8_0INT8Range(dst, weightData, scales, input, batch, inDim, outDim, start, end)
		})
	}

	if m.workers == nil || len(tasks) < m.parallelismCfg.MinTileCountForMatmulParallel {
		for _, task := range tasks {
			task()
		}
		return
	}

	m.runTasks(tasks...)
}

// runAttentionINT8 runs attention with INT8 activations
func (m *Model) runAttentionINT8(ws *modelWorkspace, output []float32, hiddenINT8 *kernels.QuantizedTensorINT8, layer *LayerINT8, seqLen int) {
	embDim := m.config.EmbedDim
	nHeads := m.config.NumHeads
	nKVHeads := m.config.NumKVHeads
	headDim := m.config.HeadDim
	kvDim := nKVHeads * headDim

	attn := &ws.attention

	// Reuse buffers for Q, K, V projections
	q := ensureFloat32(&attn.q, seqLen*embDim)
	k := ensureFloat32(&attn.k, seqLen*kvDim)
	v := ensureFloat32(&attn.v, seqLen*kvDim)

	// Project using INT8 x Q8_0 matmul (raw bytes, zero-copy)
	// Use fixed tile size for good parallelism (don't use HeadDim which may be too large)
	const tileSizeHint = 64 // Optimal for most models, provides good parallel granularity

	m.matMulQ8_0INT8Tiled(q, layer.qWeightQ8, layer.qWeightScales, hiddenINT8, seqLen, embDim, embDim, tileSizeHint)
	m.matMulQ8_0INT8Tiled(k, layer.kWeightQ8, layer.kWeightScales, hiddenINT8, seqLen, embDim, kvDim, tileSizeHint)
	m.matMulQ8_0INT8Tiled(v, layer.vWeightQ8, layer.vWeightScales, hiddenINT8, seqLen, embDim, kvDim, tileSizeHint)

	// Q/K normalization (FP32)
	// Parallelize across heads when head count >= threshold to amortize task overhead.
	// Each task processes ~4-8 heads to balance granularity vs dispatch cost.
	if len(layer.qNormWeight) > 0 {
		headsPerTask := 4
		if nHeads >= 16 {
			headsPerTask = 8
		}
		taskCount := (nHeads + headsPerTask - 1) / headsPerTask
		tasks := make([]func(), 0, taskCount)

		for hStart := 0; hStart < nHeads; hStart += headsPerTask {
			start := hStart
			end := start + headsPerTask
			if end > nHeads {
				end = nHeads
			}

			tasks = append(tasks, func() {
				for s := 0; s < seqLen; s++ {
					for h := start; h < end; h++ {
						offset := s*embDim + h*headDim
						kernels.RMSNorm(q[offset:offset+headDim], q[offset:offset+headDim], layer.qNormWeight, m.config.NormEps)
					}
				}
			})
		}

		// Only parallelize if we have enough heads (adaptive threshold from config)
		// Threshold ensures at least 2 tasks to justify dispatch overhead
		minTasks := 2
		if nHeads < m.parallelismCfg.MinHeadsForQKNormParallel {
			minTasks = 99999 // Force serial execution below threshold
		}
		m.runTasksThreshold(tasks, minTasks)
	}

	if len(layer.kNormWeight) > 0 {
		headsPerTask := 4
		if nKVHeads >= 16 {
			headsPerTask = 8
		}
		taskCount := (nKVHeads + headsPerTask - 1) / headsPerTask
		tasks := make([]func(), 0, taskCount)

		for hStart := 0; hStart < nKVHeads; hStart += headsPerTask {
			start := hStart
			end := start + headsPerTask
			if end > nKVHeads {
				end = nKVHeads
			}

			tasks = append(tasks, func() {
				for s := 0; s < seqLen; s++ {
					for h := start; h < end; h++ {
						offset := s*kvDim + h*headDim
						kernels.RMSNorm(k[offset:offset+headDim], k[offset:offset+headDim], layer.kNormWeight, m.config.NormEps)
					}
				}
			})
		}

		// Only parallelize if we have enough heads (adaptive threshold from config)
		// Threshold ensures at least 2 tasks to justify dispatch overhead
		minTasks := 2
		if nKVHeads < m.parallelismCfg.MinHeadsForQKNormParallel {
			minTasks = 99999 // Force serial execution below threshold
		}
		m.runTasksThreshold(tasks, minTasks)
	}

	// Apply RoPE (FP32) with parallelization
	if m.config.UseRoPE {
		pos := ensureInt(&attn.positions, seqLen)
		for i := 0; i < seqLen; i++ {
			pos[i] = i
		}
		// Use parallel cached RoPE for fast position embeddings
		// Parallelizes when seqLen × nHeads >= threshold (adaptive from config)
		kernels.ApplyRoPECachedParallel(q, seqLen, nHeads, headDim, pos, m.ropeCache, m.runTasks, m.parallelismCfg.MinRoPEWorkForParallel)
		kernels.ApplyRoPECachedParallel(k, seqLen, nKVHeads, headDim, pos, m.ropeCache, m.runTasks, m.parallelismCfg.MinRoPEWorkForParallel)
	}

	// For GQA, expand K, V
	var kExpanded, vExpanded []float32
	if nKVHeads < nHeads {
		kExpandedBuf := ensureFloat32(&attn.kExpanded, seqLen*embDim)
		vExpandedBuf := ensureFloat32(&attn.vExpanded, seqLen*embDim)

		kExpanded = kExpandedBuf[:seqLen*embDim]
		vExpanded = vExpandedBuf[:seqLen*embDim]
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

	// Run attention (FP32) using reusable scratch buffers
	attnOut := ensureFloat32(&attn.attnOut, seqLen*embDim)
	attnScratch := ensureFloat32(&attn.scratch, seqLen*seqLen*nHeads)
	chunkSize := 0
	if m.workers != nil && seqLen >= m.parallelismCfg.MinSeqLenForNormParallel {
		chunkSize = seqLen / 2
		if chunkSize > 128 {
			chunkSize = 128
		}
		if chunkSize < 64 {
			chunkSize = 64
		}
		if chunkSize >= seqLen {
			chunkSize = seqLen - 1
		}
	}
	// Always use MultiHeadAttentionChunked to enable head parallelism
	// When chunkSize=0, only head parallelism is used (no chunking)
	// Head parallelism threshold is adaptive from config (default: 8 heads)
	kernels.MultiHeadAttentionChunked(attnOut, q, kExpanded, vExpanded, 1, seqLen, nHeads, headDim, nil, m.config.AttentionScale, attnScratch, chunkSize, m.runTasks, m.parallelismCfg.MinHeadsForAttentionParallel)

	// Quantize attention output
	kernels.QuantizeSymmetricINT8Into(&attn.attnOutINT8, attnOut, seqLen, embDim)

	// Project output with INT8 (raw bytes, zero-copy)
	m.matMulQ8_0INT8Tiled(output, layer.oWeightQ8, layer.oWeightScales, &attn.attnOutINT8, seqLen, embDim, embDim, tileSizeHint)
}

// runMLPINT8 runs MLP with INT8 activations
func (m *Model) runMLPINT8(ws *modelWorkspace, output []float32, hiddenINT8 *kernels.QuantizedTensorINT8, layer *LayerINT8, seqLen int) {
	embDim := m.config.EmbedDim
	intermDim := m.config.IntermDim

	// Reuse buffers for gate and up projections
	mlp := &ws.mlp
	gate := ensureFloat32(&mlp.gate, seqLen*intermDim)
	up := ensureFloat32(&mlp.up, seqLen*intermDim)

	// Gate and up projections with INT8 (raw bytes, zero-copy)
	// Use 128 for MLP since intermDim is typically larger
	const mlpTileSizeHint = 128

	m.matMulQ8_0INT8Tiled(gate, layer.gateWeightQ8, layer.gateWeightScales, hiddenINT8, seqLen, embDim, intermDim, mlpTileSizeHint)
	m.matMulQ8_0INT8Tiled(up, layer.upWeightQ8, layer.upWeightScales, hiddenINT8, seqLen, embDim, intermDim, mlpTileSizeHint)

	// Apply GELU to gate (FP32) - using SIMD-optimized version
	kernels.GELUQuickSIMD(gate, gate, seqLen*intermDim)

	// Element-wise multiply
	kernels.VecMulF32(gate, gate, up, seqLen*intermDim)

	// Quantize for down projection
	kernels.QuantizeSymmetricINT8Into(&mlp.gateINT8, gate, seqLen, intermDim)

	// Down projection with INT8 (raw bytes, zero-copy)
	const downTileSizeHint = 64
	m.matMulQ8_0INT8Tiled(output, layer.downWeightQ8, layer.downWeightScales, &mlp.gateINT8, seqLen, intermDim, embDim, downTileSizeHint)
}

// extractQ8_0Scales extracts float32 scales from raw Q8_0 quantized data
// Q8_0 format: 34 bytes per block = [2-byte float16 scale] + [32 int8 values]
// Returns one float32 scale per block
func extractQ8_0Scales(data []byte, cols int) []float32 {
	if len(data) == 0 || cols <= 0 {
		return nil
	}

	const blockBytes = 34
	blocksPerRow := (cols + 31) / 32
	bytesPerRow := blocksPerRow * blockBytes
	rows := len(data) / bytesPerRow
	if rows == 0 || blocksPerRow == 0 {
		return nil
	}

	scales := make([]float32, rows*blocksPerRow)

	for r := 0; r < rows; r++ {
		rowOffset := r * bytesPerRow
		scaleBase := r * blocksPerRow
		for b := 0; b < blocksPerRow; b++ {
			offset := rowOffset + b*blockBytes
			scaleBits := uint16(data[offset]) | uint16(data[offset+1])<<8
			scales[scaleBase+b] = float16ToFloat32(scaleBits)
		}
	}

	return scales
}

func float16ToFloat32(f16 uint16) float32 {
	sign := (f16 >> 15) & 0x1
	exponent := (f16 >> 10) & 0x1f
	fraction := f16 & 0x3ff

	if exponent == 0 {
		if fraction == 0 {
			return math.Float32frombits(uint32(sign) << 31)
		}
		// Subnormal number
		for fraction&0x400 == 0 {
			fraction <<= 1
			exponent--
		}
		exponent++
		fraction &= 0x3ff
	} else if exponent == 0x1f {
		// Inf or NaN
		return math.Float32frombits((uint32(sign) << 31) | 0x7f800000 | (uint32(fraction) << 13))
	}

	// Normalized number
	exponent32 := uint32(exponent) + (127 - 15)
	fraction32 := uint32(fraction) << 13

	return math.Float32frombits((uint32(sign) << 31) | (exponent32 << 23) | fraction32)
}

func ensureFloat32(buf *[]float32, size int) []float32 {
	if cap(*buf) < size {
		*buf = make([]float32, size)
	}
	return (*buf)[:size]
}

func ensureInt(buf *[]int, size int) []int {
	if cap(*buf) < size {
		*buf = make([]int, size)
	}
	return (*buf)[:size]
}

// applyRMSNormParallel applies RMSNorm to each token in the sequence.
// Parallelizes across tokens when seqLen >= threshold to improve throughput on long sequences.
// Each token's normalization is independent, so we can process chunks in parallel.
func (m *Model) applyRMSNormParallel(hidden []float32, normWeight []float32, seqLen, embDim int) {
	serialNorm := func() {
		for i := 0; i < seqLen; i++ {
			offset := i * embDim
			kernels.RMSNorm(
				hidden[offset:offset+embDim],
				hidden[offset:offset+embDim],
				normWeight,
				m.config.NormEps,
			)
		}
	}

	// Serial fallback for short sequences or no worker pool
	if m.workers == nil || seqLen == 0 {
		serialNorm()
		return
	}

	workerCount := goruntime.GOMAXPROCS(0)
	if workerCount <= 0 {
		workerCount = 1
	}
	if workerCount == 1 {
		serialNorm()
		return
	}

	minSeqLenForParallel := m.parallelismCfg.MinSeqLenForNormParallel
	if seqLen < minSeqLenForParallel {
		serialNorm()
		return
	}

	// Determine chunk size: target one chunk per worker but clamp so that tiny
	// sequences still get work while avoiding 1-token shards.
	chunkSize := (seqLen + workerCount - 1) / workerCount
	if chunkSize < 4 {
		chunkSize = 4
	}

	switch {
	case seqLen >= 256:
		if chunkSize > 64 {
			chunkSize = 64
		}
	case seqLen >= 128:
		if chunkSize > 48 {
			chunkSize = 48
		}
	default:
		if chunkSize > 16 {
			chunkSize = 16
		}
	}

	if chunkSize >= seqLen {
		chunkSize = seqLen
	}

	// Create tasks for each chunk
	taskCount := (seqLen + chunkSize - 1) / chunkSize
	tasks := make([]func(), 0, taskCount)

	for start := 0; start < seqLen; start += chunkSize {
		end := start + chunkSize
		if end > seqLen {
			end = seqLen
		}

		// Capture loop variables
		tokenStart := start
		tokenEnd := end

		tasks = append(tasks, func() {
			for i := tokenStart; i < tokenEnd; i++ {
				offset := i * embDim
				kernels.RMSNorm(
					hidden[offset:offset+embDim],
					hidden[offset:offset+embDim],
					normWeight,
					m.config.NormEps,
				)
			}
		})
	}

	// Use threshold-based parallelization: only dispatch if we have enough tasks
	// minTasks=2 ensures we have at least 2 chunks of work to distribute
	m.runTasksThreshold(tasks, 2)
}
