// Package main demonstrates semantic similarity with an embedded model.
//
// This example imports the "model" package which embeds the GGUF model file.
// The resulting binary is self-contained (~300MB) and requires no external files.
//
// Build: go build ./examples/similarity-embedded
// Run:   ./similarity-embedded
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/headlands-org/go-semantica"
	"github.com/headlands-org/go-semantica/model"
)

func main() {
	// Load the embedded model
	rt, err := model.Open()
	if err != nil {
		log.Fatalf("Failed to open embedded model: %v", err)
	}
	defer rt.Close()

	fmt.Printf("Embedded model loaded: %d dimensions\n\n", rt.EmbedDim())

	// Test sentences: two similar, one different
	sentences := []string{
		"The cat sits on the mat",
		"A feline rests on the rug",
		"Quantum computers use superposition",
	}

	// Generate embeddings for each sentence
	ctx := context.Background()
	embeddings, err := rt.Embed(ctx, sentences, semantica.TaskSemanticSimilarity, semantica.DimensionsDefault)
	if err != nil {
		log.Fatalf("Failed to embed sentences: %v", err)
	}
	for i, sentence := range sentences {
		fmt.Printf("Embedded [%d]: \"%s\"\n", i+1, sentence)
	}

	fmt.Println()

	// Compute pairwise cosine similarities
	var maxSim float64
	var maxI, maxJ int

	for i := 0; i < len(sentences); i++ {
		for j := i + 1; j < len(sentences); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("Similarity between sentence %d and %d: %.4f\n", i+1, j+1, sim)

			if sim > maxSim {
				maxSim = sim
				maxI, maxJ = i, j
			}
		}
	}

	fmt.Printf("\nMost similar pair:\n")
	fmt.Printf("  [%d] %s\n", maxI+1, sentences[maxI])
	fmt.Printf("  [%d] %s\n", maxJ+1, sentences[maxJ])
	fmt.Printf("  Similarity: %.4f\n", maxSim)
}

// cosineSimilarity computes the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
