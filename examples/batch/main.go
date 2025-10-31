// Package main demonstrates batch embedding generation.
// This shows how to efficiently process multiple texts in parallel using the Embed() API.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/headlands-org/go-semantica"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model.gguf>\n", os.Args[0])
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Load the model
	rt, err := semantica.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	fmt.Printf("Model loaded: %d dimensions\n\n", rt.EmbedDim())

	// Sample texts covering various topics
	texts := []string{
		"Machine learning models learn patterns from data",
		"The Eiffel Tower is located in Paris, France",
		"Photosynthesis converts sunlight into chemical energy",
		"The Pacific Ocean is the largest ocean on Earth",
		"Neural networks are inspired by biological brains",
		"Mozart composed music during the Classical period",
		"DNA contains genetic instructions for organisms",
		"The Great Wall of China is visible from space",
		"Deep learning uses multiple layers of neurons",
		"The Amazon rainforest produces 20% of Earth's oxygen",
	}

	fmt.Printf("Processing %d texts in batch...\n", len(texts))

	// Generate embeddings in batch using document-optimised prompts.
	inputs := make([]semantica.Input, len(texts))
	for i, text := range texts {
		inputs[i] = semantica.Input{
			Task:    semantica.TaskSearchDocument,
			Title:   "none",
			Content: text,
		}
	}

	ctx := context.Background()
	start := time.Now()

	embeddings, err := rt.EmbedInputs(ctx, inputs)
	if err != nil {
		log.Fatalf("Failed to generate batch embeddings: %v", err)
	}

	elapsed := time.Since(start)

	// Display results
	fmt.Printf("\nBatch processing complete!\n")
	fmt.Printf("Total time: %.2f ms\n", float64(elapsed.Milliseconds()))
	fmt.Printf("Texts processed: %d\n", len(embeddings))
	fmt.Printf("Average time per text: %.2f ms\n", float64(elapsed.Milliseconds())/float64(len(texts)))
	fmt.Printf("Throughput: %.1f texts/second\n\n", float64(len(texts))/elapsed.Seconds())

	// Show sample embedding
	fmt.Printf("Sample embedding (first text):\n")
	fmt.Printf("Text: \"%s\"\n", texts[0])
	fmt.Printf("Dimensions: %d\n", len(embeddings[0]))
	fmt.Printf("First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		embeddings[0][0], embeddings[0][1], embeddings[0][2],
		embeddings[0][3], embeddings[0][4])
}
