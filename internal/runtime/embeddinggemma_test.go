//go:build integration
// +build integration

package runtime

import (
	"fmt"
	"strings"
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

// Reference embeddings from llama.cpp for "Hello world"
// Command: llama-embedding -m embeddinggemma-300m-Q8_0.gguf -p "Hello world" --embd-normalize 2
var referenceEmbedding = []float32{
	0.045353, 0.019009, -0.023062, 0.048222, -0.015216, -0.064155, -0.010043, 0.017571,
	0.030328, 0.002995, -0.024283, -0.060874, 0.007241, 0.030302, -0.026810, 0.016176,
	0.014976, 0.004858, 0.014418, -0.058049,
	// Note: llama.cpp output shows first 20 values. We'll validate these first.
}

func TestEmbeddingGemmaFullPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// 1. Load the model
	t.Log("Loading EmbeddingGemma model...")
	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer model.Close()

	// 2. Verify model configuration
	config := model.Config()
	t.Logf("Model configuration:")
	t.Logf("  Vocab size: %d", config.VocabSize)
	t.Logf("  Embed dim: %d", config.EmbedDim)
	t.Logf("  Layers: %d", config.NumLayers)
	t.Logf("  Heads: %d", config.NumHeads)
	t.Logf("  Max seq len: %d", config.MaxSeqLen)

	if config.EmbedDim != 768 {
		t.Errorf("Expected embed dim 768, got %d", config.EmbedDim)
	}
	if config.NumLayers != 24 {
		t.Errorf("Expected 24 layers, got %d", config.NumLayers)
	}

	// 3. Tokenize input
	text := "Hello world"
	t.Logf("\nTokenizing: %q", text)
	tokenIDs, err := model.Tokenizer().Encode(text)
	if err != nil {
		t.Fatalf("Tokenization failed: %v", err)
	}
	t.Logf("  Token IDs: %v", tokenIDs)
	t.Logf("  Token count: %d", len(tokenIDs))

	// 4. Generate embedding
	t.Log("\nGenerating embedding...")
	embedding, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if len(embedding) != 768 {
		t.Errorf("Expected 768-dim embedding, got %d", len(embedding))
	}

	// 5. Compare with llama.cpp reference
	t.Log("\nComparing with llama.cpp reference:")
	t.Logf("  Our first 20 values:")
	for i := 0; i < 20 && i < len(embedding); i++ {
		t.Logf("    [%2d] %.6f", i, embedding[i])
	}

	t.Logf("\n  llama.cpp reference:")
	for i := 0; i < len(referenceEmbedding); i++ {
		t.Logf("    [%2d] %.6f", i, referenceEmbedding[i])
	}

	// 6. Calculate numerical accuracy
	maxDiff := float32(0)
	var maxDiffIdx int
	for i := 0; i < len(referenceEmbedding) && i < len(embedding); i++ {
		diff := abs(embedding[i] - referenceEmbedding[i])
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
	}

	t.Logf("\n  Max difference: %.6f at index %d", maxDiff, maxDiffIdx)
	t.Logf("    Our value:   %.6f", embedding[maxDiffIdx])
	t.Logf("    Reference:   %.6f", referenceEmbedding[maxDiffIdx])

	// 7. Calculate cosine similarity (requires full embedding from llama.cpp)
	// For now, we validate the first 20 values have small error
	tolerance := float32(0.01) // Allow 1% error for quantization differences
	if maxDiff > tolerance {
		t.Errorf("Max difference %.6f exceeds tolerance %.6f", maxDiff, tolerance)
	} else {
		t.Logf("\n✅ Numerical accuracy validated (max diff %.6f < %.6f)", maxDiff, tolerance)
	}
}

func TestEmbeddingGemmaModelLoading(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Log("Testing EmbeddingGemma model loading with runtime...")

	reader, err := gguf.Open(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer reader.Close()

	// Extract configuration from metadata
	config := &ModelConfig{}

	// Architecture
	if arch, ok := reader.GetMetadata("general.architecture"); ok {
		t.Logf("Architecture: %v", arch)
		if arch != "gemma-embedding" {
			t.Errorf("Expected gemma-embedding, got %v", arch)
		}
	}

	// Embedding dimension
	if embDim, ok := reader.GetMetadata("gemma-embedding.embedding_length"); ok {
		config.EmbedDim = int(embDim.(uint32))
		t.Logf("Embedding dimension: %d", config.EmbedDim)
	}

	// Number of layers
	if layers, ok := reader.GetMetadata("gemma-embedding.block_count"); ok {
		config.NumLayers = int(layers.(uint32))
		t.Logf("Layers: %d", config.NumLayers)
	}

	// Number of attention heads
	if heads, ok := reader.GetMetadata("gemma-embedding.attention.head_count"); ok {
		config.NumHeads = int(heads.(uint32))
		t.Logf("Attention heads: %d", config.NumHeads)
	}

	// Head dimension
	if headDim, ok := reader.GetMetadata("gemma-embedding.attention.head_count_kv"); ok {
		// Note: This might not be head_dim directly, may need calculation
		config.HeadDim = int(headDim.(uint32))
		t.Logf("Head dimension (KV heads): %d", config.HeadDim)
	}

	// Vocabulary size
	if tokens, ok := reader.GetMetadata("tokenizer.ggml.tokens"); ok {
		if tokArr, ok := tokens.([]interface{}); ok {
			config.VocabSize = len(tokArr)
			t.Logf("Vocabulary size: %d", config.VocabSize)
		}
	}

	// Max sequence length
	if maxSeq, ok := reader.GetMetadata("gemma-embedding.context_length"); ok {
		config.MaxSeqLen = int(maxSeq.(uint32))
		t.Logf("Max sequence length: %d", config.MaxSeqLen)
	}

	// RoPE settings
	if ropeBase, ok := reader.GetMetadata("gemma-embedding.rope.freq_base"); ok {
		config.RoPEBase = ropeBase.(float32)
		t.Logf("RoPE frequency base: %.1f", config.RoPEBase)
	}

	// Norm epsilon
	if normEps, ok := reader.GetMetadata("gemma-embedding.attention.layer_norm_rms_epsilon"); ok {
		config.NormEps = normEps.(float32)
		t.Logf("Norm epsilon: %e", config.NormEps)
	}

	// Verify we can load key tensors
	t.Log("\nChecking key tensors:")

	keyTensors := []string{
		"token_embd.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.attn_norm.weight",
		"blk.0.post_attention_norm.weight",
	}

	for _, name := range keyTensors {
		desc, exists := reader.GetTensor(name)
		if exists {
			t.Logf("  ✓ %s: %s %v", name, desc.DType, desc.Shape)
		} else {
			t.Logf("  ✗ %s: NOT FOUND", name)
		}
	}
}

func TestEmbeddingGemmaTensorAnalysis(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := gguf.Open(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer reader.Close()

	// Analyze all tensor names to understand the architecture
	tensorNames := reader.ListTensors()

	t.Logf("Analyzing %d tensors...\n", len(tensorNames))

	// Group tensors by layer
	layers := make(map[int][]string)
	otherTensors := []string{}

	for _, name := range tensorNames {
		if strings.Contains(name, "blk.") {
			// Extract layer number
			parts := strings.Split(name, ".")
			if len(parts) >= 2 {
				var layerNum int
				if _, err := fmt.Sscanf(parts[1], "%d", &layerNum); err == nil {
					layers[layerNum] = append(layers[layerNum], name)
				}
			}
		} else {
			otherTensors = append(otherTensors, name)
		}
	}

	t.Logf("Found %d layer groups", len(layers))
	t.Logf("Found %d non-layer tensors", len(otherTensors))

	// Show first layer's tensors as example
	if tensors, ok := layers[0]; ok {
		t.Log("\nLayer 0 tensors:")
		for _, name := range tensors {
			desc, _ := reader.GetTensor(name)
			t.Logf("  %s: %s %v", name, desc.DType, desc.Shape)
		}
	}

	// Show non-layer tensors
	t.Log("\nNon-layer tensors:")
	for _, name := range otherTensors {
		desc, _ := reader.GetTensor(name)
		t.Logf("  %s: %s %v", name, desc.DType, desc.Shape)
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
