// +build integration

package runtime

import (
	"testing"
)

func TestDebugTokenEmbeddings(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Get token IDs
	tokenIDs := []int{2, 9259, 1902, 1}
	embDim := model.config.EmbedDim

	t.Log("Token embeddings (first 10 values):")
	for i, tokenID := range tokenIDs {
		offset := tokenID * embDim
		t.Logf("Token %d (ID=%d):", i, tokenID)
		for j := 0; j < 10; j++ {
			t.Logf("  [%d] = %.6f", j, model.tokenEmbed[offset+j])
		}
	}

	// Manually compute scaled embeddings
	scaleFactor := float32(27.71281)  // sqrt(768)
	t.Logf("\nInput scaling factor: %.5f (sqrt(%d))", scaleFactor, embDim)

	// Check first token embedding after scaling
	t.Log("\nFirst token embedding after scaling (first 10 values):")
	for j := 0; j < 10; j++ {
		scaled := model.tokenEmbed[2*embDim+j] * scaleFactor
		t.Logf("  [%d] = %.6f", j, scaled)
	}
}

func TestDebugNormWeights(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Check first layer normalization weights
	layer := &model.layers[0]

	t.Log("Layer 0 attnNormWeight (first 10 values):")
	for i := 0; i < 10; i++ {
		t.Logf("  [%d] = %.6f", i, layer.attnNormWeight[i])
	}

	if len(layer.qNormWeight) > 0 {
		t.Log("\nLayer 0 qNormWeight (first 10 values):")
		for i := 0; i < 10; i++ {
			t.Logf("  [%d] = %.6f", i, layer.qNormWeight[i])
		}
	}

	if len(layer.kNormWeight) > 0 {
		t.Log("\nLayer 0 kNormWeight (first 10 values):")
		for i := 0; i < 10; i++ {
			t.Logf("  [%d] = %.6f", i, layer.kNormWeight[i])
		}
	}

	t.Log("\nModel config:")
	t.Logf("  AttentionScale: %.6f", model.config.AttentionScale)
	t.Logf("  HeadDim: %d", model.config.HeadDim)
	t.Logf("  NumHeads: %d", model.config.NumHeads)
	t.Logf("  NumKVHeads: %d", model.config.NumKVHeads)
}
