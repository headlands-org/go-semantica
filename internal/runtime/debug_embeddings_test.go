// +build integration

package runtime

import (
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
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
	bytesPerRow := ((embDim + 31) / 32) * 34

	t.Log("Token embeddings (first 10 values):")
	embedding := make([]float32, embDim)
	for i, tokenID := range tokenIDs {
		// Dequantize Q8_0 embedding on-the-fly
		rowOffset := tokenID * bytesPerRow
		rowData := model.tokenEmbedQ8[rowOffset : rowOffset+bytesPerRow]
		gguf.DequantizeQ8_0Row(embedding, rowData, embDim)

		t.Logf("Token %d (ID=%d):", i, tokenID)
		for j := 0; j < 10; j++ {
			t.Logf("  [%d] = %.6f", j, embedding[j])
		}
	}

	// Manually compute scaled embeddings
	scaleFactor := float32(27.71281)  // sqrt(768)
	t.Logf("\nInput scaling factor: %.5f (sqrt(%d))", scaleFactor, embDim)

	// Check token ID=2 embedding after scaling
	rowOffset := 2 * bytesPerRow
	rowData := model.tokenEmbedQ8[rowOffset : rowOffset+bytesPerRow]
	gguf.DequantizeQ8_0Row(embedding, rowData, embDim)

	t.Log("\nToken ID=2 embedding after scaling (first 10 values):")
	for j := 0; j < 10; j++ {
		scaled := embedding[j] * scaleFactor
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

	// INT8 layers are loaded lazily - trigger load
	_, err = model.ForwardINT8([]int{2})
	if err != nil {
		t.Fatalf("ForwardINT8 failed: %v", err)
	}

	// Check first layer normalization weights from INT8 layer
	layer := model.layersINT8[0]

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
