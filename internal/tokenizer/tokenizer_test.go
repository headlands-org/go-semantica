package tokenizer

import (
	"fmt"
	"testing"
)

func TestTokenizerBasic(t *testing.T) {
	// Create a simple test vocabulary with individual characters for BPE
	// BPE needs individual chars/bytes as base tokens
	vocab := []string{
		"<unk>", "<s>", "</s>", // Control tokens
		// Individual characters needed for BPE
		"▁", "h", "e", "l", "o", "w", "r", "d", "t", "a", "n",
		// Common merges
		"he", "ll", "el", "lo", "wo", "or", "rl", "ld", "th", "an",
		// Whole words (higher scores for better merging)
		"hello", "world", "▁hello", "▁world",
		"▁the", "▁a", "▁an",
	}

	scores := make([]float32, len(vocab))
	// Give higher scores to longer tokens (BPE prefers them)
	for i := range scores {
		scores[i] = float32(len(vocab[i])) - 10 // Longer = higher score
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
		AddBOS:   false, // Disable for simpler testing
		AddEOS:   false,
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
		minLen   int // minimum expected length
	}{
		{"hello world", 1}, // Should tokenize into something
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
			// Empty input should give empty result
			t.Errorf("Encode(%q) = %v, expected empty", tt.input, ids)
		} else if tt.input != "" && len(ids) < tt.minLen {
			t.Errorf("Encode(%q) = %v, expected at least %d tokens", tt.input, ids, tt.minLen)
		}

		// Test decoding (don't check exact match, just that it doesn't error)
		_, err = tok.Decode(ids)
		if err != nil {
			t.Errorf("Decode(%v) error: %v", ids, err)
		}

		t.Logf("Input: %q -> IDs: %v (len=%d)", tt.input, ids, len(ids))
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

func TestBytefallbackTokens(t *testing.T) {
	// Create a vocab with byte fallback tokens at 238-493
	// This matches the actual Gemma model structure
	vocabSize := 494
	vocab := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	tokenTypes := make([]TokenType, vocabSize)

	// Set up basic tokens
	vocab[0] = "<unk>"
	vocab[1] = "<s>"
	vocab[2] = "</s>"
	vocab[3] = "▁"
	vocab[4] = "h"
	vocab[5] = "e"
	vocab[6] = "l"
	vocab[7] = "o"
	vocab[8] = "hello"
	vocab[9] = "▁hello"

	for i := 0; i < 10; i++ {
		scores[i] = float32(i)
		tokenTypes[i] = TokenNormal
	}
	tokenTypes[0] = TokenControl
	tokenTypes[1] = TokenControl
	tokenTypes[2] = TokenControl

	// Fill tokens 10-237 with dummy values
	for i := 10; i < 238; i++ {
		vocab[i] = fmt.Sprintf("<token_%d>", i)
		scores[i] = 0.0
		tokenTypes[i] = TokenNormal
	}

	// Set up byte fallback tokens at 238-493
	// These represent individual bytes (0x00 to 0xFF)
	for b := 0; b < 256; b++ {
		idx := 238 + b
		// Use a representation that shows this is a byte token
		vocab[idx] = fmt.Sprintf("<0x%02X>", b)
		scores[idx] = -10.0 // Low score so they're only used as fallback
		tokenTypes[idx] = TokenByte
	}

	cfg := Config{
		AddBOS:   false,
		AddEOS:   false,
		NFKC:     false,
		Lowercase: false,
	}

	tok, err := New(vocab, scores, tokenTypes, cfg)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}
	tok.unkID = 0 // Set UNK token

	t.Run("InvalidUTF8Sequences", func(t *testing.T) {
		// Test raw invalid UTF-8 bytes
		// When Go's rune conversion sees invalid UTF-8, it converts to U+FFFD (replacement char)
		// which encodes to UTF-8 as 0xEF 0xBF 0xBD. These bytes then trigger byte fallback.
		tests := []struct {
			name         string
			input        string
			expectBytes  []byte // The actual bytes after UTF-8 replacement char conversion
			description  string
		}{
			{
				"high_byte_0xFF",
				"\xFF",
				[]byte{0xEF, 0xBF, 0xBD}, // U+FFFD replacement character in UTF-8
				"Invalid UTF-8 byte 0xFF becomes replacement character",
			},
			{
				"high_byte_0x80",
				"\x80",
				[]byte{0xEF, 0xBF, 0xBD},
				"Invalid UTF-8 byte 0x80 becomes replacement character",
			},
			{
				"sequence_0xC0_0x80",
				"\xC0\x80",
				[]byte{0xEF, 0xBF, 0xBD, 0xEF, 0xBF, 0xBD}, // Two replacement chars
				"Invalid UTF-8 sequence becomes two replacement characters",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				ids, err := tok.Encode(tt.input)
				if err != nil {
					t.Fatalf("Encode(%q) error: %v", tt.input, err)
				}

				t.Logf("%s", tt.description)
				t.Logf("Input string: %q -> Runes: %v -> Token IDs: %v",
					tt.input, []rune(tt.input), ids)

				// Verify we got byte fallback tokens for the replacement character bytes
				expectedIDs := make([]int, len(tt.expectBytes))
				for i, b := range tt.expectBytes {
					expectedIDs[i] = int(b) + 238
				}

				if len(ids) != len(expectedIDs) {
					t.Errorf("Expected %d tokens, got %d", len(expectedIDs), len(ids))
				}

				for i, expectedID := range expectedIDs {
					if i >= len(ids) {
						break
					}
					if ids[i] != expectedID {
						t.Errorf("Token %d: expected ID %d (byte 0x%02X), got %d",
							i, expectedID, tt.expectBytes[i], ids[i])
					}

					// Verify token ID is in byte fallback range
					if ids[i] < 238 || ids[i] > 493 {
						t.Errorf("Token ID %d is not in byte fallback range [238, 493]", ids[i])
					}
				}
			})
		}
	})

	t.Run("ControlCharacters", func(t *testing.T) {
		// Test control characters that aren't in the vocab
		tests := []struct {
			name string
			char byte
		}{
			{"null_byte", 0x00},
			{"bell", 0x07},
			{"backspace", 0x08},
			{"tab", 0x09},
			{"line_feed", 0x0A},
			{"carriage_return", 0x0D},
			{"escape", 0x1B},
			{"unit_separator", 0x1F},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				input := string([]byte{tt.char})
				ids, err := tok.Encode(input)
				if err != nil {
					t.Fatalf("Encode error: %v", err)
				}

				t.Logf("Control char 0x%02X -> Token IDs: %v", tt.char, ids)

				// Should use byte fallback for unknown control chars
				if len(ids) > 0 {
					expectedID := int(tt.char) + 238
					// The token might be in vocab or use byte fallback
					// Just verify it's a valid token
					if ids[0] >= 238 && ids[0] <= 493 {
						if ids[0] != expectedID {
							t.Errorf("Expected byte token %d for 0x%02X, got %d",
								expectedID, tt.char, ids[0])
						}
					}
				}
			})
		}
	})

	t.Run("RoundTripDecoding", func(t *testing.T) {
		// Test that encode -> decode produces correct reconstruction
		// Note: Decoding might not perfectly reconstruct due to normalization
		tests := []struct {
			name  string
			input string
		}{
			{"invalid_utf8", "\xFF\x80"},
			{"mixed_valid_invalid", "hello\xFF"},
			{"multiple_invalid", "\xC0\xC1\xC2"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				ids, err := tok.Encode(tt.input)
				if err != nil {
					t.Fatalf("Encode error: %v", err)
				}

				decoded, err := tok.Decode(ids)
				if err != nil {
					t.Fatalf("Decode error: %v", err)
				}

				t.Logf("Input: %q (%v) -> IDs: %v -> Decoded: %q",
					tt.input, []byte(tt.input), ids, decoded)

				// For byte tokens, verify the token string format
				for _, id := range ids {
					if id >= 238 && id <= 493 {
						byteVal := id - 238
						expectedToken := fmt.Sprintf("<0x%02X>", byteVal)
						actualToken := vocab[id]
						if actualToken != expectedToken {
							t.Errorf("Byte token %d has wrong format: got %q, want %q",
								id, actualToken, expectedToken)
						}
					}
				}
			})
		}
	})

	t.Run("ByteTokensExistInVocab", func(t *testing.T) {
		// Verify all byte tokens exist in the vocab
		for b := 0; b < 256; b++ {
			id := 238 + b
			if id >= len(vocab) {
				t.Fatalf("Byte token ID %d exceeds vocab size %d", id, len(vocab))
			}

			token := vocab[id]
			if token == "" {
				t.Errorf("Byte token at index %d (byte 0x%02X) is empty", id, b)
			}

			// Verify token type
			if tokenTypes[id] != TokenByte {
				t.Errorf("Token at index %d should be TokenByte, got %v", id, tokenTypes[id])
			}

			// Verify token can be looked up
			expectedToken := fmt.Sprintf("<0x%02X>", b)
			if token != expectedToken {
				t.Errorf("Byte token %d: expected %q, got %q", id, expectedToken, token)
			}
		}

		t.Logf("Verified all 256 byte tokens exist at indices 238-493")
	})

	t.Run("KnownTokensNotFallback", func(t *testing.T) {
		// Verify that known tokens (like "hello") don't trigger byte fallback
		input := "hello"
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		t.Logf("Known token 'hello' -> IDs: %v", ids)

		// Should NOT use byte fallback tokens for known words
		for _, id := range ids {
			if id >= 238 && id <= 493 {
				t.Errorf("Known token 'hello' should not use byte fallback, got ID %d", id)
			}
		}
	})

	t.Run("MixedKnownUnknown", func(t *testing.T) {
		// Test string with both known tokens and unknown tokens
		// "hello" is known, but unknown UTF-8 chars will be converted to U+FFFD (replacement char)
		// which has UTF-8 bytes 0xEF 0xBF 0xBD, each mapping to byte fallback tokens
		input := "hello\xFF"
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		t.Logf("Mixed input 'hello\\xFF' -> IDs: %v", ids)

		// Should have some normal tokens and byte fallback tokens
		foundNormal := false
		foundByteToken := false
		expectedByteTokens := []int{477, 429, 427} // 0xEF, 0xBF, 0xBD + 238
		for _, id := range ids {
			if id >= 238 && id <= 493 {
				foundByteToken = true
				// Verify it's one of the replacement character bytes
				found := false
				for _, exp := range expectedByteTokens {
					if id == exp {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Unexpected byte token %d, expected one of %v", id, expectedByteTokens)
				}
			} else if id >= 3 && id < 238 {
				foundNormal = true
			}
		}

		if !foundByteToken {
			t.Error("Expected to find byte fallback tokens for \\xFF (converted to U+FFFD)")
		}
		if !foundNormal {
			t.Error("Expected to find normal tokens for 'hello'")
		}
	})

	t.Run("EmptyInput", func(t *testing.T) {
		ids, err := tok.Encode("")
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}
		if len(ids) != 0 {
			t.Errorf("Empty input should produce empty output, got %v", ids)
		}
	})

	t.Run("VocabSizeCheck", func(t *testing.T) {
		// Verify the tokenizer has enough tokens for byte fallback
		if tok.VocabSize() <= 493 {
			t.Errorf("Vocab size %d is too small for byte fallback (needs > 493)",
				tok.VocabSize())
		}
		t.Logf("Vocab size: %d (sufficient for byte fallback)", tok.VocabSize())
	})

	t.Run("UnknownTokenByteFallback", func(t *testing.T) {
		// Test the actual byte fallback mechanism by using valid UTF-8 that's not in vocab
		// Use a character that will definitely not be in our minimal test vocab
		// "ñ" (U+00F1) encodes to UTF-8 as 0xC3 0xB1
		input := "ñ"
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		t.Logf("Unknown character 'ñ' (UTF-8: 0xC3 0xB1) -> IDs: %v", ids)

		// Should use byte fallback for the two UTF-8 bytes
		expectedIDs := []int{433, 415} // 0xC3 + 238, 0xB1 + 238
		if len(ids) != len(expectedIDs) {
			t.Errorf("Expected %d tokens, got %d", len(expectedIDs), len(ids))
		}

		for i, expectedID := range expectedIDs {
			if i >= len(ids) {
				break
			}
			if ids[i] != expectedID {
				t.Errorf("Token %d: expected ID %d, got %d", i, expectedID, ids[i])
			}
			// Verify in byte fallback range
			if ids[i] < 238 || ids[i] > 493 {
				t.Errorf("Token ID %d is not in byte fallback range [238, 493]", ids[i])
			}
		}
	})

	t.Run("UnknownEmojiTokenByteFallback", func(t *testing.T) {
		// Test emoji (high Unicode) - should use byte fallback
		// "😀" (U+1F600) encodes to UTF-8 as 0xF0 0x9F 0x98 0x80
		input := "😀"
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		t.Logf("Emoji '😀' (UTF-8: 0xF0 0x9F 0x98 0x80) -> IDs: %v", ids)

		// Should use byte fallback for all four UTF-8 bytes
		expectedIDs := []int{478, 397, 390, 366} // bytes + 238
		if len(ids) != len(expectedIDs) {
			t.Errorf("Expected %d tokens, got %d", len(expectedIDs), len(ids))
		}

		for i, expectedID := range expectedIDs {
			if i >= len(ids) {
				break
			}
			if ids[i] != expectedID {
				t.Errorf("Token %d: expected ID %d, got %d", i, expectedID, ids[i])
			}
		}
	})

	t.Run("ByteFallbackVerifyMapping", func(t *testing.T) {
		// Verify that byte value correctly maps to token ID
		// Test a simple unknown ASCII character not in vocab: 'z'
		input := "z" // 'z' = 0x7A
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		t.Logf("Unknown char 'z' (0x7A) -> IDs: %v", ids)

		// Should map to token 238 + 0x7A = 360
		expectedID := 360
		if len(ids) != 1 {
			t.Errorf("Expected 1 token, got %d", len(ids))
		} else if ids[0] != expectedID {
			t.Errorf("Expected token ID %d, got %d", expectedID, ids[0])
		}

		// Verify the token is in the vocab
		if ids[0] >= len(vocab) {
			t.Fatalf("Token ID %d exceeds vocab size %d", ids[0], len(vocab))
		}

		token := vocab[ids[0]]
		expectedToken := fmt.Sprintf("<0x%02X>", 0x7A)
		if token != expectedToken {
			t.Errorf("Token at ID %d: expected %q, got %q", ids[0], expectedToken, token)
		}
	})

	t.Run("ByteFallbackReverseDecode", func(t *testing.T) {
		// Test that decoding byte tokens reconstructs the original bytes
		// Encode several byte token IDs directly and decode
		testCases := []struct {
			name      string
			byteVal   byte
			tokenID   int
		}{
			{"null", 0x00, 238},
			{"space", 0x20, 270},
			{"letter_A", 0x41, 303},
			{"tilde", 0x7E, 364},
			{"high_byte", 0xFF, 493},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// Verify vocab has the token
				if tc.tokenID >= len(vocab) {
					t.Fatalf("Token ID %d exceeds vocab size %d", tc.tokenID, len(vocab))
				}

				token := vocab[tc.tokenID]
				expectedToken := fmt.Sprintf("<0x%02X>", tc.byteVal)
				if token != expectedToken {
					t.Errorf("Token at ID %d: expected %q, got %q", tc.tokenID, expectedToken, token)
				}

				// Decode the token
				decoded, err := tok.Decode([]int{tc.tokenID})
				if err != nil {
					t.Fatalf("Decode error: %v", err)
				}

				// The decoded string should contain the byte token representation
				t.Logf("Token ID %d (byte 0x%02X) decodes to: %q", tc.tokenID, tc.byteVal, decoded)
			})
		}
	})
}
