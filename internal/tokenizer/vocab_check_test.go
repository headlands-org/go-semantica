// +build integration

package tokenizer

import (
	"strings"
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func TestVocabTokenStrings(t *testing.T) {
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

	// Check what the actual vocab strings are
	testIDs := []int{26352, 1902, 9259, 12392, 640, 535, 844}

	t.Log("Actual vocabulary token strings:")
	for _, id := range testIDs {
		if id < len(tok.vocab) {
			token := tok.vocab[id]
			// Show the token with escaped characters
			escaped := strings.ReplaceAll(token, "▁", "\\u2581")
			t.Logf("  ID %d: %q (escaped: %s)", id, token, escaped)
		}
	}

	// Now check what we're searching for
	text := "Hello world"
	preprocessed := "▁" + strings.ReplaceAll(text, " ", "▁")
	t.Logf("\nPreprocessed text: %q", preprocessed)

	// Check if substrings match
	t.Log("\nChecking if expected tokens are in preprocessed text:")
	expectedTokens := []string{"▁Hello", "▁world", "▁H", "el", "lo"}
	for _, expected := range expectedTokens {
		if strings.Contains(preprocessed, expected) {
			if id, ok := tok.tokenToID[expected]; ok {
				t.Logf("  %q: FOUND in text, FOUND in vocab (ID %d, score %.2f)",
					expected, id, tok.scores[id])
			} else {
				t.Logf("  %q: FOUND in text, NOT in vocab", expected)
			}
		} else {
			t.Logf("  %q: NOT in text", expected)
		}
	}
}
