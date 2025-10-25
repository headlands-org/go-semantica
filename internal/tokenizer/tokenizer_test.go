package tokenizer

import (
	"testing"
)

func TestTokenizerBasic(t *testing.T) {
	// Create a simple test vocabulary
	vocab := []string{
		"<unk>", "<s>", "</s>",
		"hello", "world", "▁hello", "▁world",
		"▁the", "▁a", "▁an",
		"!", "?", ".",
	}

	scores := make([]float32, len(vocab))
	for i := range scores {
		scores[i] = -float32(i) // Higher index = lower score
	}

	tokenTypes := make([]TokenType, len(vocab))
	for i := range tokenTypes {
		if i < 3 {
			tokenTypes[i] = TokenControl
		} else {
			tokenTypes[i] = TokenNormal
		}
	}

	cfg := Config{
		AddBOS:   true,
		AddEOS:   true,
		NFKC:     false,
		Lowercase: false,
	}

	tok, err := New(vocab, scores, tokenTypes, cfg)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Test encoding
	tests := []struct {
		input    string
		minLen   int // minimum expected length (with BOS/EOS)
	}{
		{"hello world", 2},
		{"the", 1},
		{"", 0},
	}

	for _, tt := range tests {
		ids, err := tok.Encode(tt.input)
		if err != nil {
			t.Errorf("Encode(%q) error: %v", tt.input, err)
			continue
		}

		if tt.input == "" && len(ids) > 0 {
			// Empty input should give empty or just special tokens
			t.Logf("Encode(%q) = %v", tt.input, ids)
		} else if tt.input != "" && len(ids) < tt.minLen {
			t.Errorf("Encode(%q) = %v, expected at least %d tokens", tt.input, ids, tt.minLen)
		}

		// Test decoding
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Errorf("Decode(%v) error: %v", ids, err)
		}

		t.Logf("Input: %q -> IDs: %v -> Decoded: %q", tt.input, ids, decoded)
	}
}

func TestNormalizer(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		lowercase bool
		nfkc      bool
		wantDiff  bool // expect different from input
	}{
		{"no-op", "hello", false, false, false},
		{"lowercase", "Hello", true, false, true},
		{"nfkc", "ﬁ", false, true, true}, // ligature fi -> f i
		{"both", "Ｈｅｌｌｏ", true, true, true}, // fullwidth -> ascii + lowercase
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			norm := NewNormalizer(tt.lowercase, false, tt.nfkc)
			result := norm.Normalize(tt.input)

			if tt.wantDiff && result == tt.input {
				t.Errorf("Expected normalization to change %q", tt.input)
			}
			if !tt.wantDiff && result != tt.input {
				t.Errorf("Expected normalization to preserve %q, got %q", tt.input, result)
			}

			t.Logf("Normalize(%q) = %q", tt.input, result)
		})
	}
}

func TestTokenizerVocabSize(t *testing.T) {
	vocab := []string{"a", "b", "c"}
	scores := []float32{1, 2, 3}

	tok, err := New(vocab, scores, nil, Config{})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if tok.VocabSize() != 3 {
		t.Errorf("VocabSize() = %d, expected 3", tok.VocabSize())
	}
}
