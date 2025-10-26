# Vocabulary Token Analysis for EmbeddingGemma-300M

## Summary

Analysis of vocabulary tokens in `model/embeddinggemma-300m-Q8_0.gguf` to understand tokenization behavior, especially for whitespace characters.

**Total vocabulary size:** 262,144 tokens

## Key Findings

### Special Tokens (0-106)
- Token 0: `<pad>` - Padding token
- Token 1: `<eos>` - End of sequence
- Token 2: `<bos>` - Beginning of sequence
- Token 3: `<unk>` - Unknown token
- Token 106: `<end_of_turn>` - Turn ending marker

### Whitespace Tokens

#### Multi-Space Tokens (107-140+)
Token 107 is BOTH a special newline token AND the boundary before multi-space tokens:
- **Token 107**: `"\n"` (0x0A) - Single newline character
- **Token 108**: `"\n\n"` - Double newline
- **Token 138**: `"  "` - Two spaces (0x20 0x20)
- **Token 139**: `"   "` - Three spaces (0x20 0x20 0x20)
- **Token 140**: `"    "` - Four spaces

#### Single Tab Token
- **Token 255968**: `"\t"` (0x09) - Tab character

### Byte Fallback Tokens (238-493)

The model uses byte fallback tokens at indices 238-493 (256 tokens total) for encoding individual bytes that don't have dedicated vocabulary entries:

- **Token 238**: `<0x00>` - Byte 0x00
- **Token 247**: `<0x09>` - Byte 0x09 (tab as byte)
- **Token 248**: `<0x0A>` - Byte 0x0A (newline as byte)
- **Token 270**: `<0x20>` - Byte 0x20 (space as byte)
- **Token 493**: `<0xFF>` - Byte 0xFF

**Formula:** Byte value B → Token ID = 238 + B

**Note:** This is different from the test mock vocabulary which uses range 236000-236255.

### Regular Vocabulary Tokens (494+)

Regular tokens start at index 494 and include:
- **Token 9952**: `"▁spaces"` - Word "spaces" with SentencePiece marker
- **Token 35220**: `"spaces"` - Word "spaces" without marker
- **Token 236743**: `"▁"` - SentencePiece space marker (U+2581, bytes: 0xE2 0x96 0x81)
- **Token 236746**: `"a"` - Single letter 'a'

The `▁` character (U+2581 "Lower One Eighth Block") is used by SentencePiece to mark word boundaries where spaces were removed during tokenization.

## Test Case Analysis

### Test Case: "  spaces  "
**Expected tokens:** `[2, 138, 35220, 138, 1]`

Breakdown:
1. Token 2 = `<bos>` (beginning of sequence)
2. Token 138 = `"  "` (two leading spaces)
3. Token 35220 = `"spaces"` (the word without space marker)
4. Token 138 = `"  "` (two trailing spaces)
5. Token 1 = `<eos>` (end of sequence)

### Test Case: "new\nlines"
**Expected tokens:** `[2, 1951, 107, 8721, 1]`

Breakdown:
1. Token 2 = `<bos>`
2. Token 1951 = `"new"` (the word)
3. Token 107 = `"\n"` (newline character)
4. Token 8721 = `"lines"` (the word)
5. Token 1 = `<eos>`

### Test Case: "tab\there"
**Expected tokens:** `[2, 4823, 255968, 8472, 1]`

Breakdown:
1. Token 2 = `<bos>`
2. Token 4823 = `"tab"` (the word)
3. Token 255968 = `"\t"` (tab character)
4. Token 8472 = `"here"` (the word)
5. Token 1 = `<eos>`

### Test Case: "multiple   spaces"
**Expected tokens:** `[2, 43819, 139, 35220, 1]`

Breakdown:
1. Token 2 = `<bos>`
2. Token 43819 = `"multiple"` (the word)
3. Token 139 = `"   "` (three spaces)
4. Token 35220 = `"spaces"` (the word)
5. Token 1 = `<eos>`

## Token ID Ranges Summary

| Range | Purpose | Examples |
|-------|---------|----------|
| 0-106 | Special tokens | `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<end_of_turn>` |
| 107-137 | Newlines and control | `\n`, `\n\n` |
| 138-200+ | Multi-space tokens | Two spaces, three spaces, four spaces, etc. |
| 238-493 | Byte fallback | `<0x00>` through `<0xFF>` |
| 494-235999 | Regular vocabulary | Common words, subwords |
| 236000+ | Extended vocabulary | Rare words, Unicode, special tokens |
| 255968+ | Special whitespace | Tab character at 255968 |

## Important Notes

1. **Dual representation for whitespace:**
   - Space can be `"  "` (token 138 for two spaces) OR `<0x20>` (token 270 as byte)
   - Newline can be `"\n"` (token 107) OR `<0x0A>` (token 248 as byte)
   - Tab can be `"\t"` (token 255968) OR `<0x09>` (token 247 as byte)

2. **SentencePiece marker:**
   - The `▁` character (token 236743) is NOT a space
   - It's a visual marker showing where a word-initial space was removed
   - Appears at the start of tokens like `"▁spaces"` (token 9952)

3. **Tokenizer behavior:**
   - The tokenizer should prefer dedicated whitespace tokens (107, 138, 139, 255968)
   - Byte fallback tokens are used for unknown characters or forced byte-level encoding
   - Multi-space sequences get their own tokens rather than multiple single-space tokens

## Implications for Tokenizer Implementation

The tokenizer must:
1. Handle multi-space sequences as single tokens (138, 139, etc.)
2. Recognize newline as token 107, not byte token 248
3. Recognize tab as token 255968, not byte token 247
4. Properly decode the SentencePiece marker `▁` when reconstructing text
5. Fall back to byte tokens (238-493) only for truly unknown sequences
