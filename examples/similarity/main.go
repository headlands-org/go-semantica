// Simple example showing semantic similarity between three sentences
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model.gguf>\n", os.Args[0])
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Load model
	fmt.Printf("Loading model from %s...\n", modelPath)
	start := time.Now()
	rt, err := ggufembed.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()
	loadTime := time.Since(start)
	fmt.Printf("Model loaded in %v\n\n", loadTime)

	// Three sentences: two similar (about AI/ML), one different (about food)
	sentences := []string{
		"Machine learning algorithms can recognize patterns in data",
		"Neural networks are powerful tools for pattern recognition",
		"My favorite pizza topping is pepperoni and mushrooms",
	}

	fmt.Println("Generating embeddings for:")
	for i, s := range sentences {
		fmt.Printf("  [%d] %q\n", i, s)
	}
	fmt.Println()

	// Generate embeddings and measure time
	embedStart := time.Now()
	embeddings := make([][]float32, len(sentences))
	for i, text := range sentences {
		embedding, err := rt.EmbedSingle(context.Background(), text)
		if err != nil {
			log.Fatalf("Failed to generate embedding for sentence %d: %v", i, err)
		}
		embeddings[i] = embedding
	}
	embedTime := time.Since(embedStart)

	fmt.Printf("Generated %d embeddings in %v (avg: %v per embedding)\n\n",
		len(embeddings), embedTime, embedTime/time.Duration(len(embeddings)))

	// Compare all pairs and find most similar
	fmt.Println("Similarity scores:")
	maxSim := float32(-1)
	maxI, maxJ := -1, -1

	for i := 0; i < len(sentences); i++ {
		for j := i + 1; j < len(sentences); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("  [%d] <-> [%d]: %.4f\n", i, j, sim)

			if sim > maxSim {
				maxSim = sim
				maxI, maxJ = i, j
			}
		}
	}

	fmt.Printf("\nMost similar pair: [%d] and [%d] (similarity: %.4f)\n", maxI, maxJ, maxSim)
	fmt.Printf("  [%d] %q\n", maxI, sentences[maxI])
	fmt.Printf("  [%d] %q\n", maxJ, sentences[maxJ])
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
