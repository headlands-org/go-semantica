#!/bin/bash

# generate_token_references.sh
#
# Generates reference token IDs using llama.cpp's llama-tokenize tool
# for validating our Go tokenizer implementation.
#
# Prerequisites:
#   - llama-tokenize must be installed (from llama.cpp project)
#   - Model file must exist at: model/embeddinggemma-300m-Q8_0.gguf
#
# Usage:
#   ./scripts/generate_token_references.sh
#
# Output:
#   testdata/reference_token_ids.json
#
# Notes:
#   - Re-run this script whenever you need to regenerate reference token IDs
#   - The script will overwrite the existing reference file

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$PROJECT_ROOT/model/embeddinggemma-300m-Q8_0.gguf"
OUTPUT_PATH="$PROJECT_ROOT/testdata/reference_token_ids.json"

# Check prerequisites
if ! command -v llama-tokenize &> /dev/null; then
    echo "Error: llama-tokenize not found in PATH"
    echo "Please install llama.cpp: https://github.com/ggerganov/llama.cpp"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model is downloaded or pull from Git LFS:"
    echo "  git lfs pull --include=\"model/embeddinggemma-300m-Q8_0.gguf\""
    exit 1
fi

# Function to tokenize a single text
tokenize() {
    local text="$1"
    # Use --no-escape to preserve literal \n, \t etc. in the input
    # llama-tokenize outputs format: [1, 2, 3]
    llama-tokenize -m "$MODEL_PATH" -p "$text" --ids --log-disable 2>&1 | tr -d '\n'
}

echo "Generating reference token IDs using llama.cpp..."
echo "Model: $MODEL_PATH"
echo ""

# Start JSON output
cat > "$OUTPUT_PATH" << 'EOF'
{
  "generated_by": "llama-tokenize (llama.cpp)",
  "model": "embeddinggemma-300m-Q8_0.gguf",
  "test_cases": [
EOF

# Define test cases
# Format: "input_text|description"
test_cases=(
    # Basic cases
    "Hello world|Basic: simple greeting"
    "The quick brown fox|Basic: common phrase"
    "Testing tokenization|Basic: tech phrase"

    # Empty/minimal
    "|Empty: empty string"
    "a|Minimal: single letter"
    "ab|Minimal: two letters"
    " |Minimal: single space"

    # Unicode
    "Hello 👋|Unicode: emoji wave"
    "你好世界|Unicode: Chinese"
    "مرحبا|Unicode: Arabic"
    "こんにちは世界|Unicode: Japanese"
    "Привет мир|Unicode: Russian"
    "γεια σου κόσμε|Unicode: Greek"

    # Punctuation
    "Hello, world!|Punctuation: comma and exclamation"
    "What's this?|Punctuation: apostrophe and question"
    "50% off!|Punctuation: percent sign"
    "test@example.com|Punctuation: email address"
    "It's a test.|Punctuation: contraction and period"

    # Whitespace
    "  spaces  |Whitespace: leading and trailing"
    $'new\nlines|Whitespace: newline character'
    $'tab\there|Whitespace: tab character'
    "multiple   spaces|Whitespace: multiple spaces"

    # Special characters
    "#hashtag|Special: hashtag"
    "@mention|Special: at-mention"
    "\$100|Special: dollar sign"
    "C++|Special: programming language"
    "foo/bar|Special: slash"
    "key=value|Special: equals"

    # Mixed content
    "你好 world 🌍|Mixed: Chinese, English, emoji"
    "Test #1: 50% done!|Mixed: numbers, symbols, punctuation"
    "Email: user@test.com 📧|Mixed: email with emoji"

    # Numbers
    "123456|Numbers: digits"
    "3.14159|Numbers: decimal"
    "1,000,000|Numbers: with commas"

    # Long text
    "This is a longer sentence with more than thirty words to test how the tokenizer handles longer sequences of text including various punctuation marks and common English words.|Long: 30+ words"
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once, making it useful for testing.|Long: pangram with explanation"
)

# Process each test case
first=true
for test_case in "${test_cases[@]}"; do
    # Split on pipe character
    IFS='|' read -r input description <<< "$test_case"

    # Add comma if not first entry
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$OUTPUT_PATH"
    fi

    # Tokenize
    token_ids=$(tokenize "$input")

    # Escape input text for JSON using jq
    escaped_input=$(echo -n "$input" | jq -R -s '.')
    escaped_description=$(echo -n "$description" | jq -R -s '.')

    # Write JSON entry (escaped_input and escaped_description already include quotes)
    cat >> "$OUTPUT_PATH" << EOF
    {
      "input": $escaped_input,
      "description": $escaped_description,
      "token_ids": $token_ids
    }
EOF

    # Progress indicator
    echo "✓ $description"
done

# Close JSON
cat >> "$OUTPUT_PATH" << 'EOF'

  ]
}
EOF

echo ""
echo "✓ Successfully generated reference token IDs"
echo "  Output: $OUTPUT_PATH"
echo "  Test cases: ${#test_cases[@]}"
echo ""
echo "You can now use this file in your tokenizer tests to validate"
echo "that the Go implementation produces identical token IDs."
