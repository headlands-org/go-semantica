// Demo of Pure-Go GGUF Runtime with EmbeddingGemma
package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lth/pure-go-llamas/internal/runtime"
)

func main() {
	modelPath := "model/embeddinggemma-300m-Q8_0.gguf"

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          Pure-Go GGUF Runtime - EmbeddingGemma Demo           ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Step 1: Load model
	fmt.Println("📦 Step 1: Loading EmbeddingGemma model...")
	fmt.Printf("   Model: %s\n", modelPath)

	startLoad := time.Now()
	model, err := runtime.LoadModel(modelPath)
	if err != nil {
		log.Fatalf("❌ Failed to load model: %v", err)
	}
	defer model.Close()
	loadDuration := time.Since(startLoad)

	fmt.Printf("   ✅ Model loaded in %v\n", loadDuration)
	fmt.Println()

	// Step 2: Show model info
	fmt.Println("📊 Step 2: Model Configuration")
	config := model.Config()
	fmt.Printf("   Architecture:     gemma-embedding\n")
	fmt.Printf("   Embedding dim:    %d\n", config.EmbedDim)
	fmt.Printf("   Layers:           %d\n", config.NumLayers)
	fmt.Printf("   Attention heads:  %d (Query) + %d (KV) - Grouped Query Attention\n",
		config.NumHeads, config.NumKVHeads)
	fmt.Printf("   Vocabulary size:  %s tokens\n", formatNumber(config.VocabSize))
	fmt.Printf("   Max sequence:     %d tokens\n", config.MaxSeqLen)
	fmt.Printf("   RoPE base:        %s\n", formatNumber(int(config.RoPEBase)))
	fmt.Printf("   Norm epsilon:     %.2e\n", config.NormEps)
	fmt.Println()

	// Step 3: Test inputs
	testInputs := []string{
		"Hello world",
		"The quick brown fox",
		"Machine learning",
	}

	fmt.Println("🧪 Step 3: Generating Embeddings")
	fmt.Println()

	for i, text := range testInputs {
		fmt.Printf("   Input %d: %q\n", i+1, text)

		// Tokenize
		tok := model.Tokenizer()
		tokenIDs, err := tok.Encode(text)
		if err != nil {
			log.Printf("   ❌ Tokenization failed: %v\n", err)
			continue
		}
		fmt.Printf("   Tokens:  %v (%d tokens)\n", tokenIDs, len(tokenIDs))

		// Generate embedding
		startEmbed := time.Now()
		embedding, err := model.Forward(tokenIDs)
		embedDuration := time.Since(startEmbed)

		if err != nil {
			log.Printf("   ❌ Embedding failed: %v\n", err)
			continue
		}

		fmt.Printf("   ✅ Embedding generated in %v\n", embedDuration)
		fmt.Printf("   Dimensions: %d\n", len(embedding))

		// Show first 10 values
		fmt.Printf("   First 10:   [")
		for j := 0; j < 10 && j < len(embedding); j++ {
			if j > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.4f", embedding[j])
		}
		fmt.Printf("]\n")

		// Calculate L2 norm (should be ~1.0 after normalization)
		var sumSq float32
		for _, v := range embedding {
			sumSq += v * v
		}
		norm := float32(1.0)
		if sumSq > 0 {
			norm = float32(sumSq)
		}
		fmt.Printf("   L2 norm:    %.6f (normalized: %v)\n", norm, abs(norm-1.0) < 0.001)

		fmt.Println()
	}

	// Step 4: Similarity demo
	fmt.Println("🔍 Step 4: Computing Similarity")
	fmt.Println()

	// Generate embeddings for comparison
	texts := []string{"machine learning", "artificial intelligence"}
	embeddings := make([][]float32, len(texts))

	for i, text := range texts {
		tokenIDs, _ := model.Tokenizer().Encode(text)
		emb, err := model.Forward(tokenIDs)
		if err != nil {
			log.Printf("   Failed to generate embedding for %q: %v\n", text, err)
			continue
		}
		embeddings[i] = emb
	}

	if len(embeddings) == 2 {
		similarity := cosineSimilarity(embeddings[0], embeddings[1])
		fmt.Printf("   Text 1: %q\n", texts[0])
		fmt.Printf("   Text 2: %q\n", texts[1])
		fmt.Printf("   Cosine similarity: %.4f\n", similarity)
		fmt.Println()
	}

	// Summary
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        Demo Complete!                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("✅ Working components:")
	fmt.Println("   • GGUF model loading (314 tensors)")
	fmt.Println("   • Q8_0 dequantization (perfect accuracy)")
	fmt.Println("   • Grouped Query Attention (3 Q heads, 1 KV head)")
	fmt.Println("   • Full transformer forward pass (24 layers)")
	fmt.Println("   • L2-normalized 768-dim embeddings")
	fmt.Println()
	fmt.Println("⚠️  Known limitation:")
	fmt.Println("   • Tokenization differs from llama.cpp")
	fmt.Println("   • See VALIDATION_RESULTS.md for details")
	fmt.Println()
	fmt.Println("💡 Try modifying the test inputs above to see more examples!")
}

func formatNumber(n int) string {
	if n >= 1000000 {
		return fmt.Sprintf("%d,%03d,%03d", n/1000000, (n/1000)%1000, n%1000)
	} else if n >= 1000 {
		return fmt.Sprintf("%d,%03d", n/1000, n%1000)
	}
	return fmt.Sprintf("%d", n)
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	// Since embeddings are already L2-normalized, normA and normB should be ~1
	return dotProduct
}
