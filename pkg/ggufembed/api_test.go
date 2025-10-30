package ggufembed

import (
	"context"
	"fmt"
	"math"
	"testing"
)

const gemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

func TestBuildPromptMatchesModelCard(t *testing.T) {
	const content = "neural networks are powerful"

	tests := []struct {
		name    string
		input   EmbedInput
		want    string
		wantErr bool
	}{
		{
			name:  "TaskNone",
			input: EmbedInput{Task: TaskNone, Content: content},
			want:  content,
		},
		{
			name:  "Search query",
			input: EmbedInput{Task: TaskSearchQuery, Content: content},
			want:  "task: search result | query: " + content,
		},
		{
			name:  "Question answering",
			input: EmbedInput{Task: TaskQuestionAnswering, Content: content},
			want:  "task: question answering | query: " + content,
		},
		{
			name:  "Fact verification",
			input: EmbedInput{Task: TaskFactVerification, Content: content},
			want:  "task: fact checking | query: " + content,
		},
		{
			name:  "Classification",
			input: EmbedInput{Task: TaskClassification, Content: content},
			want:  "task: classification | query: " + content,
		},
		{
			name:  "Clustering",
			input: EmbedInput{Task: TaskClustering, Content: content},
			want:  "task: clustering | query: " + content,
		},
		{
			name:  "Semantic similarity",
			input: EmbedInput{Task: TaskSemanticSimilarity, Content: content},
			want:  "task: sentence similarity | query: " + content,
		},
		{
			name:  "Code retrieval",
			input: EmbedInput{Task: TaskCodeRetrieval, Content: content},
			want:  "task: code retrieval | query: " + content,
		},
		{
			name:  "Document with title",
			input: EmbedInput{Task: TaskSearchDocument, Title: "Doc", Content: content},
			want:  "title: Doc | text: " + content,
		},
		{
			name:  "Document default title",
			input: EmbedInput{Task: TaskSearchDocument, Content: content},
			want:  "title: none | text: " + content,
		},
		{
			name:  "Custom description",
			input: EmbedInput{Task: TaskSemanticSimilarity, Content: content, CustomTaskDescription: "custom"},
			want:  "task: custom | query: " + content,
		},
		{
			name:    "Missing content",
			input:   EmbedInput{Task: TaskSearchQuery},
			wantErr: true,
		},
		{
			name:    "Unknown task",
			input:   EmbedInput{Task: Task(42), Content: content},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := BuildPrompt(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("prompt mismatch: want %q, got %q", tc.want, got)
			}
		})
	}
}

// TestBasicEmbedding verifies that basic embedding generation works correctly
func TestBasicEmbedding(t *testing.T) {
	// Open runtime
	rt, err := Open(gemmaModelPath)
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
	expectedDim := DefaultEmbedDim
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

	// Verify custom dimension paths
	dimTests := []int{768, 512, 256, 128}
	for _, d := range dimTests {
		t.Run(fmt.Sprintf("Dim%d", d), func(t *testing.T) {
			emb, err := rt.EmbedSingleInput(ctx, EmbedInput{
				Task:    TaskSearchQuery,
				Content: text,
				Dim:     d,
			})
			if err != nil {
				t.Fatalf("Embed with dim %d failed: %v", d, err)
			}
			if len(emb) != d {
				t.Fatalf("Expected dimension %d, got %d", d, len(emb))
			}
			var sumSq float32
			for _, v := range emb {
				sumSq += v * v
			}
			if sumSq == 0 {
				t.Fatalf("Dimension %d embedding is zero vector", d)
			}
			norm := float32(math.Sqrt(float64(sumSq)))
			if norm < 0.99 || norm > 1.01 {
				t.Fatalf("Dimension %d embedding not normalized: %f", d, norm)
			}
		})
	}

	t.Run("UnsupportedDim", func(t *testing.T) {
		_, err := rt.EmbedSingleInput(ctx, EmbedInput{
			Task:    TaskSearchQuery,
			Content: text,
			Dim:     1024,
		})
		if err == nil {
			t.Fatal("expected error for unsupported dimension")
		}
	})
}

// TestBatchProcessing verifies that batch processing works correctly
func TestBatchProcessing(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	rt, err := Open(gemmaModelPath)
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
			inputs := make([]EmbedInput, len(tc.texts))
			for i, text := range tc.texts {
				inputs[i] = EmbedInput{Task: TaskSearchDocument, Content: text}
			}

			embeddings, err := rt.EmbedInputs(ctx, inputs)
			if err != nil {
				t.Fatalf("Failed to embed: %v", err)
			}

			// Verify we got the right number of embeddings
			if len(embeddings) != len(tc.texts) {
				t.Errorf("Expected %d embeddings, got %d", len(tc.texts), len(embeddings))
			}

			// Verify each embedding
			for i, emb := range embeddings {
				if len(emb) != DefaultEmbedDim {
					t.Errorf("Embedding %d has wrong dimension: %d (expected %d)",
						i, len(emb), DefaultEmbedDim)
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

			t.Logf("Successfully processed %d texts", len(tc.texts))
		})
	}
}

// TestExplicitThreadCount verifies that explicit thread count works
func TestExplicitThreadCount(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	// Open with explicit thread count
	rt, err := Open(gemmaModelPath, WithThreads(2))
	if err != nil {
		t.Fatalf("Failed to open runtime: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()

	// Create a batch
	inputs := make([]EmbedInput, 64)
	for i := range inputs {
		inputs[i] = EmbedInput{Task: TaskSearchDocument, Content: "Test text"}
	}

	embeddings, err := rt.EmbedInputs(ctx, inputs)
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	if len(embeddings) != len(inputs) {
		t.Errorf("Expected %d embeddings, got %d", len(inputs), len(embeddings))
	}

	t.Logf("Successfully processed %d texts with explicit thread count of 2", len(inputs))
}

// TestConsistency verifies that multiple runs produce identical results
func TestConsistency(t *testing.T) {
	rt, err := Open(gemmaModelPath)
	if err != nil {
		t.Fatalf("Failed to open runtime: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()
	testTexts := []string{
		"Hello, world!",
		"Machine learning is fascinating.",
		"The quick brown fox jumps over the lazy dog.",
	}

	for _, text := range testTexts {
		t.Run(text, func(t *testing.T) {
			// Generate embeddings twice
			emb1, err := rt.EmbedSingle(ctx, text)
			if err != nil {
				t.Fatalf("Failed to embed (first): %v", err)
			}

			emb2, err := rt.EmbedSingle(ctx, text)
			if err != nil {
				t.Fatalf("Failed to embed (second): %v", err)
			}

			// Calculate cosine similarity
			var dotProduct, norm1, norm2 float32
			for i := 0; i < len(emb1); i++ {
				dotProduct += emb1[i] * emb2[i]
				norm1 += emb1[i] * emb1[i]
				norm2 += emb2[i] * emb2[i]
			}

			cosineSim := float64(dotProduct) / (math.Sqrt(float64(norm1)) * math.Sqrt(float64(norm2)))

			// Embeddings should be identical
			if cosineSim < 0.9999 {
				t.Errorf("Embeddings differ: cosine similarity = %f (expected > 0.9999)", cosineSim)
			}

			t.Logf("Cosine similarity: %f", cosineSim)
		})
	}
}
