package main

import (
	"fmt"
	"log"
	"os"

	"github.com/headlands-org/go-semantica"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model-path> [text]\n", os.Args[0])
		os.Exit(1)
	}

	modelPath := os.Args[1]
	text := "Hello, world!"
	if len(os.Args) > 2 {
		text = os.Args[2]
	}

	// Open the model
	rt, err := semantica.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	// Get the tokenizer
	tok := rt.Tokenizer()

	fmt.Printf("Text: %q\n", text)
	fmt.Printf("Vocabulary size: %d\n\n", tok.VocabSize())

	// Encode text to token IDs
	ids, err := tok.Encode(text)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}

	fmt.Printf("Token IDs (%d tokens):\n", len(ids))
	for i, id := range ids {
		fmt.Printf("  [%d] = %d\n", i, id)
	}

	// Decode token IDs back to text
	decoded, err := tok.Decode(ids)
	if err != nil {
		log.Fatalf("Failed to decode: %v", err)
	}

	fmt.Printf("\nDecoded text: %q\n", decoded)
}
