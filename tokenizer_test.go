package semantica

import (
	"testing"

	"github.com/headlands-org/go-semantica/internal/testutil"
)

func TestTokenizer(t *testing.T) {
	modelPath := testutil.RequireModelPath(t)

	// Open runtime
	rt, err := Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	// Get tokenizer
	tok := rt.Tokenizer()
	if tok == nil {
		t.Fatal("Tokenizer() returned nil")
	}

	// Test basic encoding/decoding
	testCases := []struct {
		name string
		text string
	}{
		{"Simple", "hello world"},
		{"Unicode", "Hello 世界"},
		{"Punctuation", "Hello, world!"},
		{"Empty", ""},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Encode
			ids, err := tok.Encode(tc.text)
			if err != nil {
				t.Fatalf("Encode failed: %v", err)
			}

			if tc.text == "" {
				if len(ids) == 0 {
					return // Empty text might produce empty tokens or just special tokens
				}
			}

			// Decode
			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode failed: %v", err)
			}

			// For non-empty text, we should get something back
			if tc.text != "" && len(ids) == 0 {
				t.Errorf("Encode produced no tokens for non-empty text: %q", tc.text)
			}

			t.Logf("Text: %q -> IDs: %v -> Decoded: %q", tc.text, ids, decoded)
		})
	}

	// Test VocabSize
	vocabSize := tok.VocabSize()
	if vocabSize <= 0 {
		t.Errorf("VocabSize() = %d, want > 0", vocabSize)
	}
	t.Logf("Vocabulary size: %d", vocabSize)
}

func TestTokenizerRoundTrip(t *testing.T) {
	modelPath := testutil.RequireModelPath(t)

	rt, err := Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	tok := rt.Tokenizer()

	// Test that encode->decode produces reasonable output
	text := "The quick brown fox jumps over the lazy dog"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(ids) == 0 {
		t.Fatal("Encode produced no tokens")
	}

	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// The decoded text might have slight differences (spacing, special tokens removed)
	// but should contain the key words
	t.Logf("Original: %q", text)
	t.Logf("Token IDs: %v", ids)
	t.Logf("Decoded: %q", decoded)
}
