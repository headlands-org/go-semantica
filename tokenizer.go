package semantica

import "github.com/headlands-org/go-semantica/pkg/ggufembed"

// Tokenizer provides text tokenization capabilities using the model's vocabulary.
type Tokenizer struct {
	inner ggufembed.Tokenizer
}

// Encode tokenizes text into token IDs.
// The tokenizer applies text normalization and adds special tokens (BOS/EOS) as configured.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	return t.inner.Encode(text)
}

// Decode converts token IDs back to text.
// Special tokens are automatically filtered out during decoding.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	return t.inner.Decode(ids)
}

// VocabSize returns the total vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return t.inner.VocabSize()
}
