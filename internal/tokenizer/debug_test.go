//go:build integration
// +build integration

package tokenizer

import (
	"testing"

	"github.com/headlands-org/go-semantica/internal/gguf"
)

const gemmaModelPath = "../../model/embeddinggemma-300m-Q8_0.gguf"

func TestTokenizerDebug(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Load model
	reader, err := gguf.Open(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer reader.Close()

	// Load tokenizer
	tok, err := LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	t.Logf("Tokenizer loaded: vocab size = %d", tok.VocabSize())
	t.Logf("BOS ID: %d", tok.bosID)
	t.Logf("EOS ID: %d", tok.eosID)
	t.Logf("UNK ID: %d", tok.unkID)
	t.Logf("Add BOS: %v", tok.addBOS)
	t.Logf("Add EOS: %v", tok.addEOS)

	// Test tokenization
	text := "Hello world"
	t.Logf("\nTokenizing: %q", text)

	// First, try the internal tokenizeUnigram to see what it produces
	normalized := tok.normalizer.Normalize(text)
	t.Logf("After normalization: %q", normalized)

	rawTokens := tok.tokenizeUnigram(normalized)
	t.Logf("Raw tokens from Unigram: %v", rawTokens)

	// Now check if these tokens exist in vocab
	for i, token := range rawTokens {
		if id, ok := tok.tokenToID[token]; ok {
			t.Logf("  [%d] %q -> ID %d (score: %.2f)", i, token, id, tok.scores[id])
		} else {
			t.Logf("  [%d] %q -> NOT IN VOCAB (will become UNK)", i, token)
		}
	}

	// Full encode
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	t.Logf("\nFull token IDs: %v", ids)
	for i, id := range ids {
		token := ""
		if id >= 0 && id < len(tok.vocab) {
			token = tok.vocab[id]
		}
		decoded, _ := tok.Decode([]int{id})
		t.Logf("  [%d] ID=%d token=%q decoded=%q", i, id, token, decoded)
	}

	// Check specific tokens in vocab
	testTokens := []string{"Hello", "▁Hello", "world", "▁world", " Hello", " world", "▁", "▁hell", "▁world"}
	t.Log("\nChecking specific tokens in vocabulary:")
	for _, token := range testTokens {
		if id, ok := tok.tokenToID[token]; ok {
			t.Logf("  %q -> ID %d (score: %.2f)", token, id, tok.scores[id])
		} else {
			t.Logf("  %q -> NOT FOUND", token)
		}
	}
}
