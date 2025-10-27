// +build integration

package runtime

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

const referenceEmbeddingPath = "../../testdata/reference_embedding_full.txt"

func TestEmbeddingGemmaVsLlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Load llama.cpp reference embedding
	refEmb, err := loadReferenceEmbedding(referenceEmbeddingPath)
	if err != nil {
		t.Fatalf("Failed to load reference embedding: %v", err)
	}

	if len(refEmb) != 768 {
		t.Fatalf("Expected 768-dim reference, got %d", len(refEmb))
	}

	t.Logf("Loaded reference embedding: %d dimensions", len(refEmb))
	t.Logf("Reference first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
		refEmb[0], refEmb[1], refEmb[2], refEmb[3], refEmb[4],
		refEmb[5], refEmb[6], refEmb[7], refEmb[8], refEmb[9])

	// Load and run our model
	t.Log("\nLoading our model...")
	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("EmbeddingGemma model not available: %v", err)
	}
	defer model.Close()

	// Tokenize
	text := "Hello world"
	t.Logf("\nTokenizing: %q", text)
	tokenIDs, err := model.Tokenizer().Encode(text)
	if err != nil {
		t.Fatalf("Tokenization failed: %v", err)
	}
	t.Logf("Token IDs: %v (%d tokens)", tokenIDs, len(tokenIDs))

	// Decode tokens to see what we got
	tok := model.Tokenizer()
	t.Log("Tokens:")
	for i, id := range tokenIDs {
		token, err := tok.Decode([]int{id})
		if err == nil {
			t.Logf("  [%d] = %d %q", i, id, token)
		} else {
			t.Logf("  [%d] = %d (decode error: %v)", i, id, err)
		}
	}

	// Generate embedding
	t.Log("\nGenerating embedding...")
	ourEmb, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if len(ourEmb) != 768 {
		t.Fatalf("Expected 768-dim embedding, got %d", len(ourEmb))
	}

	t.Logf("Our first 10:       %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
		ourEmb[0], ourEmb[1], ourEmb[2], ourEmb[3], ourEmb[4],
		ourEmb[5], ourEmb[6], ourEmb[7], ourEmb[8], ourEmb[9])

	// Compute cosine similarity
	cosSim := cosineSimilarity(ourEmb, refEmb)
	t.Logf("\nCosine similarity: %.6f", cosSim)

	// Compute element-wise statistics
	var maxDiff, sumDiff, sumSqDiff float32
	var maxDiffIdx int

	for i := 0; i < len(ourEmb); i++ {
		diff := abs(ourEmb[i] - refEmb[i])
		sumDiff += diff
		sumSqDiff += diff * diff

		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
	}

	avgDiff := sumDiff / float32(len(ourEmb))
	rmsDiff := float32(math.Sqrt(float64(sumSqDiff / float32(len(ourEmb)))))

	t.Logf("\nNumerical comparison:")
	t.Logf("  Max difference:  %.6f at index %d", maxDiff, maxDiffIdx)
	t.Logf("    Our value:     %.6f", ourEmb[maxDiffIdx])
	t.Logf("    Ref value:     %.6f", refEmb[maxDiffIdx])
	t.Logf("  Avg difference:  %.6f", avgDiff)
	t.Logf("  RMS difference:  %.6f", rmsDiff)

	// For embedding models, cosine similarity > 0.95 is usually acceptable
	// Even > 0.90 can be fine depending on the use case
	if cosSim < 0.90 {
		t.Errorf("Cosine similarity %.6f is too low (expected > 0.90)", cosSim)
		t.Logf("\nThis suggests a potential issue with:")
		t.Logf("  1. Tokenization differences (we got %d tokens, llama.cpp used 4)", len(tokenIDs))
		t.Logf("  2. Model architecture implementation")
		t.Logf("  3. Attention or normalization layers")
	} else if cosSim < 0.99 {
		t.Logf("\n⚠️  Cosine similarity %.6f is acceptable but not perfect", cosSim)
		t.Logf("   Likely due to:")
		t.Logf("    - Tokenization differences")
		t.Logf("    - Minor implementation differences")
	} else {
		t.Logf("\n✅ Excellent agreement! Cosine similarity %.6f", cosSim)
	}
}

func loadReferenceEmbedding(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var values []float32
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		val, err := strconv.ParseFloat(strings.TrimSpace(scanner.Text()), 32)
		if err != nil {
			return nil, fmt.Errorf("parse float: %w", err)
		}
		values = append(values, float32(val))
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return values, nil
}
