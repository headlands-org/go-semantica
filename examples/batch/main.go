// Batch embedding example demonstrating efficient processing of 100 sentences
package main

import (
	"context"
	"fmt"
	"log"
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

	// Generate 100 diverse sentences
	sentences := generateSentences(100)
	fmt.Printf("Generating embeddings for %d sentences...\n", len(sentences))

	// Batch embed with timing
	embedStart := time.Now()
	embeddings, err := rt.Embed(context.Background(), sentences)
	if err != nil {
		log.Fatalf("Failed to generate batch embeddings: %v", err)
	}
	embedTime := time.Since(embedStart)

	// Report statistics
	fmt.Printf("\nBatch embedding completed:\n")
	fmt.Printf("  Total sentences: %d\n", len(embeddings))
	fmt.Printf("  Embedding dimension: %d\n", len(embeddings[0]))
	fmt.Printf("  Total time: %v\n", embedTime)
	fmt.Printf("  Average time per sentence: %v\n", embedTime/time.Duration(len(embeddings)))
	fmt.Printf("  Throughput: %.2f sentences/second\n", float64(len(embeddings))/embedTime.Seconds())

	// Show sample embeddings
	fmt.Printf("\nSample embeddings:\n")
	for i := 0; i < 3 && i < len(sentences); i++ {
		fmt.Printf("  [%d] %q\n", i, sentences[i])
		fmt.Printf("      First 5 dimensions: ")
		for j := 0; j < 5 && j < len(embeddings[i]); j++ {
			fmt.Printf("%.4f ", embeddings[i][j])
		}
		fmt.Println()
	}
}

// generateSentences creates 100 diverse sentences for testing
func generateSentences(n int) []string {
	// Template sentences with variety
	templates := []string{
		"The study of %s reveals fascinating insights about the natural world",
		"Scientists have discovered new methods for %s research",
		"Understanding %s is crucial for advancing modern technology",
		"Recent breakthroughs in %s have transformed our perspective",
		"The application of %s has led to significant innovations",
		"Researchers are exploring the connections between %s and other fields",
		"The future of %s looks promising with emerging technologies",
		"Experts predict that %s will revolutionize the industry",
		"New tools for %s analysis are improving accuracy",
		"The impact of %s on society continues to grow",
	}

	topics := []string{
		"artificial intelligence", "machine learning", "quantum physics",
		"molecular biology", "climate science", "neuroscience",
		"astrophysics", "genetics", "renewable energy", "robotics",
		"biochemistry", "computer vision", "natural language processing",
		"data science", "cryptography", "materials science",
		"cognitive science", "evolutionary biology", "oceanography",
		"nanotechnology",
	}

	sentences := make([]string, 0, n)
	for i := 0; i < n; i++ {
		template := templates[i%len(templates)]
		topic := topics[i%len(topics)]
		sentences = append(sentences, fmt.Sprintf(template, topic))
	}

	return sentences
}
