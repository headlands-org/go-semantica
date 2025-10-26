package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: vocab-search <model.gguf> <pattern>")
	}

	modelPath := os.Args[1]
	pattern := os.Args[2]

	reader, err := gguf.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer reader.Close()

	// Get vocabulary
	tokensRaw, ok := reader.GetMetadata("tokenizer.ggml.tokens")
	if !ok {
		log.Fatal("tokenizer.ggml.tokens not found in metadata")
	}

	tokensArr, ok := tokensRaw.([]interface{})
	if !ok {
		log.Fatal("tokenizer.ggml.tokens is not an array")
	}

	vocab := make([]string, len(tokensArr))
	for i, t := range tokensArr {
		vocab[i], ok = t.(string)
		if !ok {
			log.Fatalf("Token %d is not a string", i)
		}
	}

	fmt.Printf("Total vocabulary size: %d\n", len(vocab))
	fmt.Printf("Searching for pattern: %s\n\n", pattern)

	count := 0
	for i, token := range vocab {
		if strings.Contains(token, pattern) {
			count++
			if count <= 50 { // Show first 50 matches
				fmt.Printf("Token %6d: %q\n", i, token)
			}
		}
	}

	fmt.Printf("\nTotal matches: %d\n", count)
	if count > 50 {
		fmt.Printf("(showing first 50)\n")
	}
}
