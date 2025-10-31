//go:build integration
// +build integration

package gguf

import (
	"path/filepath"
	"testing"
)

const testModelPath = "../../models/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf"

func TestRealModelLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Try to open the model
	reader, err := Open(testModelPath)
	if err != nil {
		t.Skipf("Test model not available: %v", err)
	}
	defer reader.Close()

	// Verify header
	header := reader.Header()
	if header.Magic != GGUFMagic {
		t.Errorf("Invalid magic: got 0x%08x, want 0x%08x", header.Magic, GGUFMagic)
	}

	if header.Version != GGUFVersion {
		t.Errorf("Invalid version: got %d, want %d", header.Version, GGUFVersion)
	}

	t.Logf("Model loaded successfully")
	t.Logf("  Version: %d", header.Version)
	t.Logf("  Tensors: %d", header.TensorCount)
	t.Logf("  Metadata KVs: %d", header.MetadataKVSize)

	// Check metadata
	if arch, ok := reader.GetMetadata("general.architecture"); ok {
		t.Logf("  Architecture: %v", arch)
	}

	if name, ok := reader.GetMetadata("general.name"); ok {
		t.Logf("  Name: %v", name)
	}

	if embDim, ok := reader.GetMetadata("embedding.length"); ok {
		t.Logf("  Embedding dimension: %v", embDim)
	} else if embDim, ok := reader.GetMetadata("bert.embedding_length"); ok {
		t.Logf("  Embedding dimension (BERT): %v", embDim)
	}

	// Check tensors
	tensors := reader.ListTensors()
	if len(tensors) != int(header.TensorCount) {
		t.Errorf("Tensor count mismatch: got %d, want %d", len(tensors), header.TensorCount)
	}

	// Verify we can access tensor data
	if len(tensors) > 0 {
		tensorName := tensors[0]
		desc, ok := reader.GetTensor(tensorName)
		if !ok {
			t.Fatalf("Failed to get tensor: %s", tensorName)
		}

		data, err := reader.GetTensorData(tensorName)
		if err != nil {
			t.Fatalf("Failed to get tensor data: %v", err)
		}

		if int64(len(data)) != desc.Size {
			t.Errorf("Tensor data size mismatch: got %d, want %d", len(data), desc.Size)
		}

		t.Logf("Successfully accessed tensor '%s': %s %v (%d bytes)",
			tensorName, desc.DType, desc.Shape, desc.Size)
	}

	// Check for Q8_0 tensors specifically
	q8Count := 0
	for _, name := range tensors {
		desc, _ := reader.GetTensor(name)
		if desc.DType == DTypeQ8_0 {
			q8Count++
		}
	}

	t.Logf("Found %d Q8_0 quantized tensors", q8Count)

	if q8Count == 0 {
		t.Error("Expected to find at least one Q8_0 tensor")
	}
}

func TestQ8_0Dequantization(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := Open(testModelPath)
	if err != nil {
		t.Skipf("Test model not available: %v", err)
	}
	defer reader.Close()

	// Find a Q8_0 tensor
	var q8Tensor string
	for _, name := range reader.ListTensors() {
		desc, _ := reader.GetTensor(name)
		if desc.DType == DTypeQ8_0 {
			q8Tensor = name
			break
		}
	}

	if q8Tensor == "" {
		t.Skip("No Q8_0 tensor found")
	}

	// Get tensor data
	desc, _ := reader.GetTensor(q8Tensor)
	data, err := reader.GetTensorData(q8Tensor)
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}

	// Dequantize
	totalElems := 1
	for _, dim := range desc.Shape {
		totalElems *= dim
	}

	dequantized := DequantizeQ8_0(data, totalElems)

	if len(dequantized) != totalElems {
		t.Errorf("Dequantization size mismatch: got %d, want %d", len(dequantized), totalElems)
	}

	// Basic sanity checks
	hasNonZero := false
	hasPositive := false
	hasNegative := false

	for _, val := range dequantized {
		if val != 0 {
			hasNonZero = true
		}
		if val > 0 {
			hasPositive = true
		}
		if val < 0 {
			hasNegative = true
		}
	}

	if !hasNonZero {
		t.Error("All dequantized values are zero")
	}

	t.Logf("Dequantized %d elements from %s", len(dequantized), filepath.Base(q8Tensor))
	t.Logf("  Has positive values: %v", hasPositive)
	t.Logf("  Has negative values: %v", hasNegative)
	t.Logf("  Sample values: %.4f, %.4f, %.4f", dequantized[0], dequantized[len(dequantized)/2], dequantized[len(dequantized)-1])
}

func TestTokenizerFromGGUF(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := Open(testModelPath)
	if err != nil {
		t.Skipf("Test model not available: %v", err)
	}
	defer reader.Close()

	// Check tokenizer metadata
	if tokModel, ok := reader.GetMetadata("tokenizer.ggml.model"); ok {
		t.Logf("Tokenizer model: %v", tokModel)
	}

	if tokens, ok := reader.GetMetadata("tokenizer.ggml.tokens"); ok {
		if tokensArr, ok := tokens.([]interface{}); ok {
			t.Logf("Vocabulary size: %d", len(tokensArr))

			// Show a few sample tokens
			for i := 0; i < 5 && i < len(tokensArr); i++ {
				t.Logf("  Token %d: %q", i, tokensArr[i])
			}
		}
	}

	if scores, ok := reader.GetMetadata("tokenizer.ggml.scores"); ok {
		if scoresArr, ok := scores.([]interface{}); ok {
			t.Logf("Scores count: %d", len(scoresArr))
		}
	}
}
