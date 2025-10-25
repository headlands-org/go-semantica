// +build integration

package tokenizer

import (
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func TestCheckSpecificVocabIDs(t *testing.T) {
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

	// Check llama.cpp's tokens
	llamaCppTokens := []int{2, 9259, 1902, 1}
	t.Log("llama.cpp tokens for 'Hello world':")
	for i, id := range llamaCppTokens {
		if id < len(tok.vocab) {
			t.Logf("  [%d] ID=%d token=%q score=%.2f", i, id, tok.vocab[id], tok.scores[id])
		}
	}

	// Check what our BOS/EOS IDs map to
	t.Logf("\nOur special token IDs:")
	t.Logf("  BOS ID=%d token=%q", tok.bosID, tok.vocab[tok.bosID])
	t.Logf("  EOS ID=%d token=%q", tok.eosID, tok.vocab[tok.eosID])
	t.Logf("  UNK ID=%d token=%q", tok.unkID, tok.vocab[tok.unkID])

	// Check what tokens are at IDs 2 and 1
	t.Logf("\nToken IDs 1 and 2:")
	t.Logf("  ID=1 token=%q", tok.vocab[1])
	t.Logf("  ID=2 token=%q", tok.vocab[2])
}
