// Package tokenizer provides SentencePiece/Unigram tokenization
package tokenizer

import (
	"fmt"
	"math"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// TokenType represents the type of a token
type TokenType int32

const (
	TokenNormal TokenType = 1
	TokenUnknown TokenType = 2
	TokenControl TokenType = 3
	TokenUserDefined TokenType = 4
	TokenUnused TokenType = 5
	TokenByte TokenType = 6
)

// SpecialToken represents a special token (BOS, EOS, UNK, etc.)
type SpecialToken struct {
	ID    int
	Token string
}

// Tokenizer is a SentencePiece/Unigram tokenizer
type Tokenizer struct {
	vocab       []string
	scores      []float32
	tokenTypes  []TokenType
	tokenToID   map[string]int
	bosID       int
	eosID       int
	unkID       int
	padID       int
	addBOS      bool
	addEOS      bool
	normalizer  Normalizer
}

// Config holds tokenizer configuration
type Config struct {
	AddBOS     bool
	AddEOS     bool
	Lowercase  bool
	RemoveAccents bool
	NFKC       bool
}

// New creates a new tokenizer
func New(vocab []string, scores []float32, tokenTypes []TokenType, cfg Config) (*Tokenizer, error) {
	if len(vocab) != len(scores) {
		return nil, fmt.Errorf("vocab and scores length mismatch: %d != %d", len(vocab), len(scores))
	}

	if len(tokenTypes) > 0 && len(vocab) != len(tokenTypes) {
		return nil, fmt.Errorf("vocab and tokenTypes length mismatch: %d != %d", len(vocab), len(tokenTypes))
	}

	// If no token types provided, assume all normal
	if len(tokenTypes) == 0 {
		tokenTypes = make([]TokenType, len(vocab))
		for i := range tokenTypes {
			tokenTypes[i] = TokenNormal
		}
	}

	t := &Tokenizer{
		vocab:      vocab,
		scores:     scores,
		tokenTypes: tokenTypes,
		tokenToID:  make(map[string]int),
		bosID:      -1,
		eosID:      -1,
		unkID:      -1,
		padID:      -1,
		addBOS:     cfg.AddBOS,
		addEOS:     cfg.AddEOS,
		normalizer: NewNormalizer(cfg.Lowercase, cfg.RemoveAccents, cfg.NFKC),
	}

	// Build reverse map
	for i, token := range vocab {
		t.tokenToID[token] = i
	}

	// Note: Special token IDs should be set from metadata, not by string matching
	// The New() function is for manual construction. LoadFromGGUF() handles metadata.

	return t, nil
}

// Encode tokenizes text to token IDs
func (t *Tokenizer) Encode(text string) ([]int, error) {
	// Normalize
	text = t.normalizer.Normalize(text)

	// SentencePiece preprocessing: replace spaces with meta-space character
	// NOTE: This simple replacement doesn't match llama.cpp's behavior for consecutive spaces.
	// llama.cpp has special vocab tokens for multiple spaces (token 138 = "  ", 139 = "   ")
	// and appears to handle them BEFORE metaspace conversion. This causes minor differences
	// in edge cases with multiple consecutive spaces, but works correctly for normal text.
	text = strings.ReplaceAll(text, " ", "▁")

	// Tokenize using BPE
	tokens := t.tokenizeBPE(text)

	// Convert to IDs
	ids := make([]int, 0, len(tokens)+2)

	if t.addBOS && t.bosID >= 0 {
		ids = append(ids, t.bosID)
	}

	for _, token := range tokens {
		if id, ok := t.tokenToID[token]; ok {
			ids = append(ids, id)
		} else if t.unkID >= 0 {
			ids = append(ids, t.unkID)
		}
	}

	if t.addEOS && t.eosID >= 0 {
		ids = append(ids, t.eosID)
	}

	return ids, nil
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var result strings.Builder

	for _, id := range ids {
		if id < 0 || id >= len(t.vocab) {
			continue
		}

		token := t.vocab[id]

		// Skip special tokens
		if t.isSpecialToken(id) {
			continue
		}

		// Replace sentencepiece underscore with space
		token = strings.ReplaceAll(token, "▁", " ")

		result.WriteString(token)
	}

	return result.String(), nil
}

// tokenizeBPE performs Byte Pair Encoding tokenization
// Based on llama.cpp's SPM tokenizer implementation
func (t *Tokenizer) tokenizeBPE(text string) []string {
	if text == "" {
		return nil
	}

	// Step 1: Split into UTF-8 characters as initial symbols
	type symbol struct {
		text  string
		prev  int
		next  int
	}

	runes := []rune(text)
	symbols := make([]symbol, len(runes))

	for i, r := range runes {
		symbols[i] = symbol{
			text: string(r),
			prev: i - 1,
			next: i + 1,
		}
	}
	if len(symbols) > 0 {
		symbols[len(symbols)-1].next = -1
	}

	// Step 2: Build priority queue of bigrams
	type bigram struct {
		left  int
		right int
		score float32
		text  string
	}

	// Helper to try adding a bigram to candidates
	tryAddBigram := func(left, right int, candidates *[]bigram) {
		if left == -1 || right == -1 {
			return
		}

		text := symbols[left].text + symbols[right].text
		if id, ok := t.tokenToID[text]; ok {
			// This bigram exists in vocab, add to candidates
			*candidates = append(*candidates, bigram{
				left:  left,
				right: right,
				score: t.scores[id],
				text:  text,
			})
		}
	}

	// Step 3: Iteratively merge highest-scoring bigrams
	for {
		// Find all valid bigrams
		var candidates []bigram
		for i := 0; i < len(symbols); i++ {
			if symbols[i].text != "" && symbols[i].next != -1 {
				tryAddBigram(i, symbols[i].next, &candidates)
			}
		}

		if len(candidates) == 0 {
			break // No more merges possible
		}

		// Find highest-scoring bigram (least negative = best)
		// When scores are equal, use rightmost position (matches llama.cpp behavior)
		bestIdx := 0
		bestScore := candidates[0].score
		for i := 1; i < len(candidates); i++ {
			if candidates[i].score > bestScore ||
				(candidates[i].score == bestScore && candidates[i].left > candidates[bestIdx].left) {
				bestScore = candidates[i].score
				bestIdx = i
			}
		}

		// Merge the best bigram
		best := candidates[bestIdx]
		left := best.left
		right := best.right

		// Merge right into left
		symbols[left].text = best.text
		symbols[left].next = symbols[right].next
		if symbols[right].next != -1 {
			symbols[symbols[right].next].prev = left
		}

		// Remove right symbol
		symbols[right].text = ""
	}

	// Step 4: Collect remaining symbols
	var result []string
	for i := 0; i < len(symbols); i++ {
		if symbols[i].text != "" {
			result = append(result, symbols[i].text)
		}
	}

	// Step 5: Resegment any remaining unknown tokens
	var final []string
	for _, token := range result {
		if _, ok := t.tokenToID[token]; ok {
			final = append(final, token)
		} else {
			// Token not in vocab, try byte tokens or use UNK
			// Byte tokens are at indices 238-493 (one per byte value 0x00-0xFF)
			hasByteTokens := len(t.vocab) > 493
			if hasByteTokens {
				// Output as byte tokens: byte value B → token index 238+B
				for _, b := range []byte(token) {
					byteTokenID := 238 + int(b)
					if byteTokenID < len(t.vocab) {
						final = append(final, t.vocab[byteTokenID])
					}
				}
			} else {
				// No byte tokens available, use UNK token
				if t.unkID >= 0 && t.unkID < len(t.vocab) {
					final = append(final, t.vocab[t.unkID])
				}
			}
		}
	}

	return final
}

// tokenizeUnigram performs Unigram tokenization using Viterbi algorithm
// DEPRECATED: Kept for reference, but BPE is now used
func (t *Tokenizer) tokenizeUnigram(text string) []string {
	if text == "" {
		return nil
	}

	runes := []rune(text)
	n := len(runes)

	// Dynamic programming: best[i] = (score, previous position)
	type state struct {
		score float64
		prev  int
		token string
	}

	best := make([]state, n+1)
	for i := range best {
		best[i].score = math.Inf(-1) // Initialize to -inf (we're maximizing)
	}
	best[0].score = 0

	// Forward pass - find best segmentation
	// Scores are negative log probabilities: higher (less negative) = better probability
	// So we MAXIMIZE the score to find the highest probability path
	for i := 0; i < n; i++ {
		if math.IsInf(best[i].score, -1) {
			continue
		}

		// Try all possible tokens starting at position i
		for j := i + 1; j <= n; j++ {
			token := string(runes[i:j])

			// Check if token exists in vocab
			id, ok := t.tokenToID[token]

			if ok {
				score := best[i].score + float64(t.scores[id])
				if score > best[j].score { // Maximize score (less negative is better)
					best[j].score = score
					best[j].prev = i
					best[j].token = token
				}
			}
		}

		// Handle unknown characters with byte fallback
		if i < n && math.IsInf(best[i+1].score, -1) {
			// Single character as unknown
			char := string(runes[i])
			score := best[i].score - 10.0 // Penalty for unknown (subtract to make more negative)
			if score > best[i+1].score {
				best[i+1].score = score
				best[i+1].prev = i
				best[i+1].token = char
			}
		}
	}

	// Backward pass - reconstruct tokens
	tokens := make([]string, 0)
	pos := n
	for pos > 0 {
		if best[pos].prev == pos {
			// Shouldn't happen, but prevent infinite loop
			break
		}
		if best[pos].token != "" {
			tokens = append(tokens, best[pos].token)
		}
		pos = best[pos].prev
	}

	// Reverse tokens (we built them backwards)
	for i := 0; i < len(tokens)/2; i++ {
		tokens[i], tokens[len(tokens)-1-i] = tokens[len(tokens)-1-i], tokens[i]
	}

	return tokens
}

// isSpecialToken checks if a token ID is a special token
func (t *Tokenizer) isSpecialToken(id int) bool {
	return id == t.bosID || id == t.eosID || id == t.unkID || id == t.padID ||
		t.tokenTypes[id] == TokenControl || t.tokenTypes[id] == TokenUnused
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}

// Normalizer handles text normalization
type Normalizer struct {
	lowercase     bool
	removeAccents bool
	nfkc          bool
}

// NewNormalizer creates a new normalizer
func NewNormalizer(lowercase, removeAccents, nfkc bool) Normalizer {
	return Normalizer{
		lowercase:     lowercase,
		removeAccents: removeAccents,
		nfkc:          nfkc,
	}
}

// Normalize normalizes text
func (n Normalizer) Normalize(text string) string {
	// NFKC normalization
	if n.nfkc {
		text = norm.NFKC.String(text)
	}

	// Remove accents
	if n.removeAccents {
		text = n.removeAccentsFunc(text)
	}

	// Lowercase
	if n.lowercase {
		text = strings.ToLower(text)
	}

	return text
}

// removeAccentsFunc removes diacritical marks
func (n Normalizer) removeAccentsFunc(s string) string {
	// Decompose to NFD
	t := norm.NFD.String(s)

	// Filter out combining marks
	var result strings.Builder
	result.Grow(len(t))

	for _, r := range t {
		if !unicode.Is(unicode.Mn, r) {
			result.WriteRune(r)
		}
	}

	return result.String()
}

// LoadFromGGUF loads a tokenizer from GGUF metadata
func LoadFromGGUF(getMetadata func(string) (interface{}, bool)) (*Tokenizer, error) {
	// Extract tokens
	tokensRaw, ok := getMetadata("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found")
	}

	tokensArr, ok := tokensRaw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not an array")
	}

	tokens := make([]string, len(tokensArr))
	for i, t := range tokensArr {
		tokens[i], ok = t.(string)
		if !ok {
			return nil, fmt.Errorf("token %d is not a string", i)
		}
	}

	// Extract scores
	scoresRaw, ok := getMetadata("tokenizer.ggml.scores")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.scores not found")
	}

	scoresArr, ok := scoresRaw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.scores is not an array")
	}

	scores := make([]float32, len(scoresArr))
	for i, s := range scoresArr {
		switch v := s.(type) {
		case float32:
			scores[i] = v
		case float64:
			scores[i] = float32(v)
		default:
			return nil, fmt.Errorf("score %d is not a number", i)
		}
	}

	// Extract token types (optional)
	var tokenTypes []TokenType
	if typesRaw, ok := getMetadata("tokenizer.ggml.token_type"); ok {
		if typesArr, ok := typesRaw.([]interface{}); ok {
			tokenTypes = make([]TokenType, len(typesArr))
			for i, t := range typesArr {
				switch v := t.(type) {
				case int32:
					tokenTypes[i] = TokenType(v)
				case uint32:
					tokenTypes[i] = TokenType(v)
				case int:
					tokenTypes[i] = TokenType(v)
				default:
					tokenTypes[i] = TokenNormal
				}
			}
		}
	}

	// Check for normalization flags in metadata
	cfg := Config{
		AddBOS: getBoolMetadata(getMetadata, "tokenizer.ggml.add_bos_token", true),
		AddEOS: getBoolMetadata(getMetadata, "tokenizer.ggml.add_eos_token", false),
		NFKC:   true, // Default for most models
	}

	tok, err := New(tokens, scores, tokenTypes, cfg)
	if err != nil {
		return nil, err
	}

	// Set special token IDs from metadata (don't rely on string matching!)
	if val, ok := getMetadata("tokenizer.ggml.bos_token_id"); ok {
		if id, ok := val.(uint32); ok {
			tok.bosID = int(id)
		}
	}

	if val, ok := getMetadata("tokenizer.ggml.eos_token_id"); ok {
		if id, ok := val.(uint32); ok {
			tok.eosID = int(id)
		}
	}

	if val, ok := getMetadata("tokenizer.ggml.unknown_token_id"); ok {
		if id, ok := val.(uint32); ok {
			tok.unkID = int(id)
		}
	}

	if val, ok := getMetadata("tokenizer.ggml.padding_token_id"); ok {
		if id, ok := val.(uint32); ok {
			tok.padID = int(id)
		}
	}

	return tok, nil
}

func getBoolMetadata(getMetadata func(string) (interface{}, bool), key string, defaultVal bool) bool {
	if val, ok := getMetadata(key); ok {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return defaultVal
}
