package runtime

import (
	"math"
	"testing"
)

const gemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

// TestDisableMatmulParallelOption verifies that the disableMatmulParallel flag
// is correctly stored in the Model struct when passed to LoadModel.
func TestDisableMatmulParallelOption(t *testing.T) {
	tests := []struct {
		name           string
		disableFlag    bool
		expectedValue  bool
	}{
		{
			name:          "Parallel enabled (default)",
			disableFlag:   false,
			expectedValue: false,
		},
		{
			name:          "Parallel disabled",
			disableFlag:   true,
			expectedValue: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Load model with the specified flag
			model, err := LoadModel(gemmaModelPath, tt.disableFlag)
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}
			defer model.Close()

			// Verify the flag is stored correctly
			if model.disableMatmulParallel != tt.expectedValue {
				t.Errorf("Expected disableMatmulParallel to be %v, got %v",
					tt.expectedValue, model.disableMatmulParallel)
			}
		})
	}
}

// TestDisableMatmulParallelBehavior verifies that the flag actually affects behavior
// by checking that inference still works with parallel disabled.
func TestDisableMatmulParallelBehavior(t *testing.T) {
	// Test with parallel enabled
	modelParallel, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		t.Fatalf("Failed to load model with parallel enabled: %v", err)
	}
	defer modelParallel.Close()

	// Test with parallel disabled
	modelSerial, err := LoadModel(gemmaModelPath, true)
	if err != nil {
		t.Fatalf("Failed to load model with parallel disabled: %v", err)
	}
	defer modelSerial.Close()

	// Test text
	testText := "Hello, world!"

	// Tokenize
	tokensParallel, err := modelParallel.Tokenizer().Encode(testText)
	if err != nil {
		t.Fatalf("Failed to tokenize (parallel): %v", err)
	}

	tokensSerial, err := modelSerial.Tokenizer().Encode(testText)
	if err != nil {
		t.Fatalf("Failed to tokenize (serial): %v", err)
	}

	// Run forward pass with parallel enabled
	embeddingParallel, err := modelParallel.Forward(tokensParallel)
	if err != nil {
		t.Fatalf("Failed to run forward pass (parallel): %v", err)
	}

	// Run forward pass with parallel disabled
	embeddingSerial, err := modelSerial.Forward(tokensSerial)
	if err != nil {
		t.Fatalf("Failed to run forward pass (serial): %v", err)
	}

	// Both should produce embeddings of the same dimension
	if len(embeddingParallel) != len(embeddingSerial) {
		t.Errorf("Embedding dimensions differ: parallel=%d, serial=%d",
			len(embeddingParallel), len(embeddingSerial))
	}

	// Embeddings should be very similar (allowing for minor floating point differences)
	// Calculate cosine similarity
	var dotProduct, normParallel, normSerial float32
	for i := 0; i < len(embeddingParallel); i++ {
		dotProduct += embeddingParallel[i] * embeddingSerial[i]
		normParallel += embeddingParallel[i] * embeddingParallel[i]
		normSerial += embeddingSerial[i] * embeddingSerial[i]
	}

	// Compute cosine similarity
	cosineSim := float64(dotProduct) / (math.Sqrt(float64(normParallel)) * math.Sqrt(float64(normSerial)))

	// Embeddings should be nearly identical (cosine similarity > 0.9999)
	if cosineSim < 0.9999 {
		t.Errorf("Embeddings differ too much: cosine similarity = %f (expected > 0.9999)", cosineSim)
	}

	t.Logf("Cosine similarity between parallel and serial: %f", cosineSim)
}
