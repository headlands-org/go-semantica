package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: vocab-inspect <model.gguf> [token_ids...]")
	}

	modelPath := os.Args[1]
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

	fmt.Printf("Total vocabulary size: %d\n\n", len(vocab))

	// If specific token IDs provided, show those
	if len(os.Args) > 2 {
		fmt.Println("Requested token details:")
		fmt.Println(strings.Repeat("-", 80))
		for _, arg := range os.Args[2:] {
			tokenID, err := strconv.Atoi(arg)
			if err != nil {
				log.Printf("Invalid token ID '%s': %v", arg, err)
				continue
			}

			if tokenID < 0 || tokenID >= len(vocab) {
				log.Printf("Token ID %d out of range [0, %d)", tokenID, len(vocab))
				continue
			}

			printTokenInfo(tokenID, vocab[tokenID])
		}
	} else {
		// Show first 20 and last 20 tokens as overview
		fmt.Println("First 20 tokens:")
		fmt.Println(strings.Repeat("-", 80))
		for i := 0; i < 20 && i < len(vocab); i++ {
			printTokenInfo(i, vocab[i])
		}

		fmt.Printf("\n... (%d tokens) ...\n\n", len(vocab)-40)

		fmt.Println("Last 20 tokens:")
		fmt.Println(strings.Repeat("-", 80))
		start := len(vocab) - 20
		if start < 20 {
			start = 20
		}
		for i := start; i < len(vocab); i++ {
			printTokenInfo(i, vocab[i])
		}
	}

	// Search for specific whitespace characters
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("Searching for whitespace tokens:")
	fmt.Println(strings.Repeat("=", 80))

	for i, token := range vocab {
		if token == "\n" {
			fmt.Printf("Token %d: NEWLINE (\\n)\n", i)
		}
		if token == "\t" {
			fmt.Printf("Token %d: TAB (\\t)\n", i)
		}
		if token == "  " {
			fmt.Printf("Token %d: TWO_SPACES\n", i)
		}
		if token == "   " {
			fmt.Printf("Token %d: THREE_SPACES\n", i)
		}
		if token == "\r" {
			fmt.Printf("Token %d: CARRIAGE_RETURN (\\r)\n", i)
		}
		if token == "\r\n" {
			fmt.Printf("Token %d: CRLF (\\r\\n)\n", i)
		}
	}
}

func printTokenInfo(id int, token string) {
	// Create readable representation
	readable := escapeString(token)

	// Show hex bytes
	hexBytes := ""
	for i, b := range []byte(token) {
		if i > 0 {
			hexBytes += " "
		}
		hexBytes += fmt.Sprintf("%02x", b)
	}

	fmt.Printf("Token %6d | %-40s | bytes: %s\n", id, readable, hexBytes)
}

func escapeString(s string) string {
	result := "\""
	for _, r := range s {
		switch r {
		case '\n':
			result += "\\n"
		case '\t':
			result += "\\t"
		case '\r':
			result += "\\r"
		case '\\':
			result += "\\\\"
		case '"':
			result += "\\\""
		case ' ':
			result += "\u2423" // Open box character for visible space
		default:
			if r < 32 || r == 127 {
				result += fmt.Sprintf("\\x%02x", r)
			} else if r > 127 {
				result += string(r) // Keep UTF-8 as is
			} else {
				result += string(r)
			}
		}
	}
	result += "\""
	return result
}
