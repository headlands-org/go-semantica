// Command gguf-inspect inspects GGUF model files
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/headlands-org/go-semantica/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model.gguf>\n", os.Args[0])
		os.Exit(1)
	}

	path := os.Args[1]

	reader, err := gguf.Open(path)
	if err != nil {
		log.Fatalf("Failed to open GGUF file: %v", err)
	}
	defer reader.Close()

	header := reader.Header()
	fmt.Printf("GGUF File: %s\n", path)
	fmt.Printf("Version: %d\n", header.Version)
	fmt.Printf("Tensor Count: %d\n", header.TensorCount)
	fmt.Printf("Metadata KV Count: %d\n\n", header.MetadataKVSize)

	// Print important metadata
	fmt.Println("=== Metadata ===")
	printMetadata(reader, "general.architecture")
	printMetadata(reader, "general.name")
	printMetadata(reader, "general.file_type")
	printMetadata(reader, "tokenizer.ggml.model")
	printMetadata(reader, "tokenizer.ggml.tokens")
	printMetadata(reader, "tokenizer.ggml.scores")
	printMetadata(reader, "tokenizer.ggml.token_type")
	fmt.Println()

	// Print tensors
	fmt.Println("=== Tensors ===")
	tensors := reader.ListTensors()
	fmt.Printf("Total: %d tensors\n\n", len(tensors))

	for i, name := range tensors {
		if i >= 10 { // Limit output
			fmt.Printf("... and %d more tensors\n", len(tensors)-10)
			break
		}

		desc, _ := reader.GetTensor(name)
		fmt.Printf("%-50s  dtype=%-8s  shape=%v  size=%d bytes\n",
			name, desc.DType, desc.Shape, desc.Size)
	}
}

func printMetadata(r *gguf.Reader, key string) {
	if val, ok := r.GetMetadata(key); ok {
		switch v := val.(type) {
		case []interface{}:
			if key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.scores" || key == "tokenizer.ggml.token_type" {
				fmt.Printf("%-30s: [%d items]\n", key, len(v))
			} else {
				fmt.Printf("%-30s: %v\n", key, v)
			}
		default:
			fmt.Printf("%-30s: %v\n", key, v)
		}
	}
}
