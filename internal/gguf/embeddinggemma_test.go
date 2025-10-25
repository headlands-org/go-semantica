// +build integration

package gguf

import (
	"strings"
	"testing"
)

const gemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

func TestEmbeddingGemmaLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := Open(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer reader.Close()

	header := reader.Header()
	t.Logf("EmbeddingGemma Model:")
	t.Logf("  Version: %d", header.Version)
	t.Logf("  Tensors: %d", header.TensorCount)
	t.Logf("  Metadata KVs: %d", header.MetadataKVSize)

	// Check architecture
	if arch, ok := reader.GetMetadata("general.architecture"); ok {
		t.Logf("  Architecture: %v", arch)
		if arch != "gemma-embedding" {
			t.Errorf("Expected gemma-embedding, got %v", arch)
		}
	} else {
		t.Error("Architecture metadata not found")
	}

	// Check embedding dimension
	if embDim, ok := reader.GetMetadata("gemma-embedding.embedding_length"); ok {
		t.Logf("  Embedding dimension: %v", embDim)
		if embDim != uint32(768) {
			t.Errorf("Expected 768, got %v", embDim)
		}
	}

	// Check layers
	if layers, ok := reader.GetMetadata("gemma-embedding.block_count"); ok {
		t.Logf("  Layers: %v", layers)
		if layers != uint32(24) {
			t.Errorf("Expected 24 layers, got %v", layers)
		}
	}

	// Check vocab size
	if tokens, ok := reader.GetMetadata("tokenizer.ggml.tokens"); ok {
		if tokArr, ok := tokens.([]interface{}); ok {
			t.Logf("  Vocabulary size: %d", len(tokArr))
			if len(tokArr) != 262144 {
				t.Errorf("Expected 262144 tokens, got %d", len(tokArr))
			}
		}
	}

	// Count Q8_0 tensors
	q8Count := 0
	f32Count := 0
	var sampleQ8Tensor string

	for _, name := range reader.ListTensors() {
		desc, _ := reader.GetTensor(name)
		if desc.DType == DTypeQ8_0 {
			if q8Count == 0 {
				sampleQ8Tensor = name
			}
			q8Count++
		} else if desc.DType == DTypeF32 {
			f32Count++
		}
	}

	t.Logf("  Q8_0 tensors: %d", q8Count)
	t.Logf("  F32 tensors: %d", f32Count)

	if q8Count == 0 {
		t.Error("No Q8_0 tensors found")
	}

	// Test Q8_0 tensor access
	if sampleQ8Tensor != "" {
		desc, _ := reader.GetTensor(sampleQ8Tensor)
		data, err := reader.GetTensorData(sampleQ8Tensor)
		if err != nil {
			t.Fatalf("Failed to get tensor data: %v", err)
		}

		t.Logf("  Sample Q8_0 tensor: %s", sampleQ8Tensor)
		t.Logf("    Shape: %v", desc.Shape)
		t.Logf("    Size: %d bytes", len(data))

		// Verify we can dequantize
		totalElems := 1
		for _, dim := range desc.Shape {
			totalElems *= dim
		}

		dequantized := DequantizeQ8_0(data, totalElems)
		if len(dequantized) != totalElems {
			t.Errorf("Dequantization size mismatch: got %d, want %d", len(dequantized), totalElems)
		}

		t.Logf("    Dequantized: %d elements", len(dequantized))
		t.Logf("    Sample values: %.4f, %.4f, %.4f",
			dequantized[0], dequantized[len(dequantized)/2], dequantized[len(dequantized)-1])
	}
}

func TestEmbeddingGemmaTensorNaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := Open(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer reader.Close()

	// Analyze tensor naming conventions
	tensorNames := reader.ListTensors()

	// Look for key tensors
	patterns := map[string]int{
		"token_embd":         0,
		"attn_q":             0,
		"attn_k":             0,
		"attn_v":             0,
		"attn_output":        0,
		"ffn_up":             0,
		"ffn_down":           0,
		"ffn_gate":           0,
		"post_attention_norm": 0,
		"attn_norm":          0,
	}

	for _, name := range tensorNames {
		for pattern := range patterns {
			if strings.Contains(name, pattern) {
				patterns[pattern]++
			}
		}
	}

	t.Log("Tensor naming analysis:")
	for pattern, count := range patterns {
		if count > 0 {
			t.Logf("  %s: %d occurrences", pattern, count)
		}
	}

	// Sample some tensor names
	t.Log("\nSample tensor names:")
	for i := 0; i < 10 && i < len(tensorNames); i++ {
		desc, _ := reader.GetTensor(tensorNames[i])
		t.Logf("  %s: %s %v", tensorNames[i], desc.DType, desc.Shape)
	}
}
