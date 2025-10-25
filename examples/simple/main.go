// Simple example of using the ggufembed package
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model.gguf>\n", os.Args[0])
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Open model
	fmt.Printf("Loading model from %s...\n", modelPath)
	rt, err := ggufembed.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	fmt.Printf("Model loaded successfully!\n")
	fmt.Printf("Embedding dimension: %d\n", rt.EmbedDim())
	fmt.Printf("Max sequence length: %d\n\n", rt.MaxSeqLen())

	// Example 1: Single embedding
	fmt.Println("=== Example 1: Single Embedding ===")
	text := "The quick brown fox jumps over the lazy dog"
	embedding, err := rt.EmbedSingle(context.Background(), text)
	if err != nil {
		log.Fatalf("Failed to generate embedding: %v", err)
	}

	fmt.Printf("Text: %q\n", text)
	fmt.Printf("Embedding: %d dimensions\n", len(embedding))
	fmt.Printf("First 10 values: ")
	for i := 0; i < 10 && i < len(embedding); i++ {
		fmt.Printf("%.4f ", embedding[i])
	}
	fmt.Println()

	// Example 2: Batch embeddings
	fmt.Println("=== Example 2: Batch Embeddings ===")
	texts := []string{
		"Machine learning is fascinating",
		"Deep learning powers modern AI",
		"The weather is nice today",
		"Go is a great programming language",
	}

	embeddings, err := rt.Embed(context.Background(), texts)
	if err != nil {
		log.Fatalf("Failed to generate batch embeddings: %v", err)
	}

	fmt.Printf("Generated %d embeddings\n\n", len(embeddings))

	// Example 3: Semantic similarity
	fmt.Println("=== Example 3: Semantic Similarity ===")
	fmt.Println("Computing pairwise similarities:")

	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("  [%d] vs [%d]: %.4f\n", i, j, sim)
		}
	}
	fmt.Println()

	// Example 4: Semantic search
	fmt.Println("=== Example 4: Semantic Search ===")
	query := "artificial intelligence"
	queryEmb, err := rt.EmbedSingle(context.Background(), query)
	if err != nil {
		log.Fatalf("Failed to generate query embedding: %v", err)
	}

	fmt.Printf("Query: %q\n", query)
	fmt.Println("Rankings:")

	type result struct {
		idx   int
		score float32
	}

	results := make([]result, len(texts))
	for i, docEmb := range embeddings {
		results[i] = result{
			idx:   i,
			score: cosineSimilarity(queryEmb, docEmb),
		}
	}

	// Sort by score (descending)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	for i, r := range results {
		fmt.Printf("  %d. [%.4f] %q\n", i+1, r.score, texts[r.idx])
	}
}

// cosineSimilarity computes the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
