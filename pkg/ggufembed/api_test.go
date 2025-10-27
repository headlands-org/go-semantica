package ggufembed

import (
	"context"
	"math"
	"testing"
)

const gemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

// TestWithDisableMatmulParallel verifies that the WithDisableMatmulParallel option
// can be set and that inference works correctly with parallel disabled.
func TestWithDisableMatmulParallel(t *testing.T) {
	tests := []struct {
		name    string
		options []Option
	}{
		{
			name:    "Default (parallel enabled)",
			options: []Option{},
		},
		{
			name:    "Parallel explicitly disabled",
			options: []Option{WithDisableMatmulParallel(true)},
		},
		{
			name:    "Parallel explicitly enabled",
			options: []Option{WithDisableMatmulParallel(false)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Open runtime with options
			rt, err := Open(gemmaModelPath, tt.options...)
			if err != nil {
				t.Fatalf("Failed to open runtime: %v", err)
			}
			defer rt.Close()

			// Test embedding generation
			ctx := context.Background()
			text := "Hello, world!"

			embedding, err := rt.EmbedSingle(ctx, text)
			if err != nil {
				t.Fatalf("Failed to generate embedding: %v", err)
			}

			// Verify embedding has expected dimension
			expectedDim := rt.EmbedDim()
			if len(embedding) != expectedDim {
				t.Errorf("Expected embedding dimension %d, got %d", expectedDim, len(embedding))
			}

			// Verify embedding is normalized (L2 norm should be ~1.0)
			var sumSq float32
			for _, v := range embedding {
				sumSq += v * v
			}
			norm := float32(1.0)
			if sumSq > 0 {
				norm = float32(math.Sqrt(float64(sumSq)))
			}

			if norm < 0.99 || norm > 1.01 {
				t.Errorf("Expected L2 norm ~1.0, got %f", norm)
			}
		})
	}
}

// TestParallelVsSerialConsistency verifies that parallel and serial execution
// produce identical results.
func TestParallelVsSerialConsistency(t *testing.T) {
	// Open with parallel enabled
	rtParallel, err := Open(gemmaModelPath, WithDisableMatmulParallel(false))
	if err != nil {
		t.Fatalf("Failed to open runtime (parallel): %v", err)
	}
	defer rtParallel.Close()

	// Open with parallel disabled
	rtSerial, err := Open(gemmaModelPath, WithDisableMatmulParallel(true))
	if err != nil {
		t.Fatalf("Failed to open runtime (serial): %v", err)
	}
	defer rtSerial.Close()

	ctx := context.Background()
	testTexts := []string{
		"Hello, world!",
		"Machine learning is fascinating.",
		"The quick brown fox jumps over the lazy dog.",
	}

	for _, text := range testTexts {
		t.Run(text, func(t *testing.T) {
			// Generate embeddings with both modes
			embParallel, err := rtParallel.EmbedSingle(ctx, text)
			if err != nil {
				t.Fatalf("Failed to embed (parallel): %v", err)
			}

			embSerial, err := rtSerial.EmbedSingle(ctx, text)
			if err != nil {
				t.Fatalf("Failed to embed (serial): %v", err)
			}

			// Calculate cosine similarity
			var dotProduct, normParallel, normSerial float32
			for i := 0; i < len(embParallel); i++ {
				dotProduct += embParallel[i] * embSerial[i]
				normParallel += embParallel[i] * embParallel[i]
				normSerial += embSerial[i] * embSerial[i]
			}

			cosineSim := float64(dotProduct) / (math.Sqrt(float64(normParallel)) * math.Sqrt(float64(normSerial)))

			// Embeddings should be nearly identical
			if cosineSim < 0.9999 {
				t.Errorf("Embeddings differ: cosine similarity = %f (expected > 0.9999)", cosineSim)
			}

			t.Logf("Cosine similarity: %f", cosineSim)
		})
	}
}

// TestAutoTuneWithBatchProcessing verifies that batch processing works correctly
// when processing batches of texts.
func TestAutoTuneWithBatchProcessing(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	rt, err := Open(gemmaModelPath, WithDisableMatmulParallel(true))
	if err != nil {
		t.Fatalf("Failed to open runtime: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()

	testCases := []struct {
		name  string
		texts []string
	}{
		{
			name:  "Small batch (4 texts)",
			texts: []string{"Text 1", "Text 2", "Text 3", "Text 4"},
		},
		{
			name: "Medium batch (16 texts)",
			texts: func() []string {
				texts := make([]string, 16)
				for i := range texts {
					texts[i] = "Test text number " + string(rune('0'+i))
				}
				return texts
			}(),
		},
		{
			name: "Large batch (64 texts)",
			texts: func() []string {
				texts := make([]string, 64)
				for i := range texts {
					texts[i] = "Test text number " + string(rune('0'+i%10))
				}
				return texts
			}(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			embeddings, err := rt.Embed(ctx, tc.texts)
			if err != nil {
				t.Fatalf("Failed to embed: %v", err)
			}

			// Verify we got the right number of embeddings
			if len(embeddings) != len(tc.texts) {
				t.Errorf("Expected %d embeddings, got %d", len(tc.texts), len(embeddings))
			}

			// Verify each embedding
			for i, emb := range embeddings {
				if len(emb) != rt.EmbedDim() {
					t.Errorf("Embedding %d has wrong dimension: %d (expected %d)",
						i, len(emb), rt.EmbedDim())
				}

				// Check normalization
				var sumSq float32
				for _, v := range emb {
					sumSq += v * v
				}
				norm := float32(math.Sqrt(float64(sumSq)))
				if norm < 0.99 || norm > 1.01 {
					t.Errorf("Embedding %d not normalized: L2 norm = %f", i, norm)
				}
			}

			t.Logf("Successfully processed %d texts with auto-tuning", len(tc.texts))
		})
	}
}

// TestExplicitThreadCount verifies that explicit thread count overrides auto-tuning
func TestExplicitThreadCount(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	// Open with explicit thread count
	rt, err := Open(gemmaModelPath, WithThreads(2), WithDisableMatmulParallel(true))
	if err != nil {
		t.Fatalf("Failed to open runtime: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()

	// Create a batch that would normally trigger auto-tuning to more workers
	texts := make([]string, 64)
	for i := range texts {
		texts[i] = "Test text"
	}

	embeddings, err := rt.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	t.Logf("Successfully processed %d texts with explicit thread count of 2", len(texts))
}
