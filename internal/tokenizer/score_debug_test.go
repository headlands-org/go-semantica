// +build integration

package tokenizer

import (
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func TestTokenScores(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := gguf.Open(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer reader.Close()

	tok, err := LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Check scores for various tokens
	testCases := []struct{
		token string
		id int
	}{
		{"▁", 236743},
		{"H", 236814},
		{"e", 236744},
		{"▁Hello", 26352},
		{"▁world", 1902},
		{"Hello", 9259},
		{"world", 12392},
	}

	t.Log("Token scores:")
	for _, tc := range testCases {
		if tc.id < len(tok.scores) {
			t.Logf("  %q (ID %d): score = %.2f", tc.token, tc.id, tok.scores[tc.id])
		}
	}

	// Calculate what the Viterbi score would be for different paths
	t.Log("\nPath scores (sum of token scores):")

	// Path 1: character by character
	charPath := []int{236743, 236814, 236744, 236752, 236752, 236748, 236743, 236765, 236748, 236750, 236752, 236753}
	charScore := 0.0
	for _, id := range charPath {
		if id < len(tok.scores) {
			charScore += float64(tok.scores[id])
		}
	}
	t.Logf("  Character-by-character: %.2f", charScore)

	// Path 2: words
	wordPath := []int{26352, 1902} // ▁Hello ▁world
	wordScore := 0.0
	for _, id := range wordPath {
		if id < len(tok.scores) {
			wordScore += float64(tok.scores[id])
		}
	}
	t.Logf("  Word tokens (▁Hello + ▁world): %.2f", wordScore)

	// Path 3: alternative
	altPath := []int{9259, 1902} // Hello + ▁world
	altScore := 0.0
	for _, id := range altPath {
		if id < len(tok.scores) {
			altScore += float64(tok.scores[id])
		}
	}
	t.Logf("  Alternative (Hello + ▁world): %.2f", altScore)

	t.Logf("\nBest path (minimum score): ")
	if charScore < wordScore && charScore < altScore {
		t.Log("  Character-by-character wins (this is what we're seeing)")
	} else if wordScore < charScore && wordScore < altScore {
		t.Log("  Word tokens should win")
	} else {
		t.Log("  Alternative should win")
	}
}
