//go:build integration

package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

// ReferenceTestCase represents a single test case from llama.cpp reference data
type ReferenceTestCase struct {
	Input       string `json:"input"`
	Description string `json:"description"`
	TokenIDs    []int  `json:"token_ids"`
}

// ReferenceData is the structure of the reference JSON file
type ReferenceData struct {
	TestCases []ReferenceTestCase `json:"test_cases"`
}

const (
	referenceDataPath = "../../testdata/reference_token_ids.json"
	modelPath         = "../../model/embeddinggemma-300m-Q8_0.gguf"
)

// TestTokenIDsVsLlamaCpp validates that our token IDs match llama.cpp exactly
// for diverse input texts. This ensures 100% tokenization parity.
func TestTokenIDsVsLlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Check if reference data exists
	if _, err := os.Stat(referenceDataPath); os.IsNotExist(err) {
		t.Skipf("Reference data file not found at %s (will be created by Task 2.2)", referenceDataPath)
	}

	// Load reference data
	referenceData, err := loadReferenceData(referenceDataPath)
	if err != nil {
		t.Fatalf("Failed to load reference data: %v", err)
	}

	if len(referenceData.TestCases) == 0 {
		t.Fatal("Reference data contains no test cases")
	}

	t.Logf("Loaded %d test cases from reference data", len(referenceData.TestCases))

	// Load model and tokenizer
	reader, err := gguf.Open(modelPath)
	if err != nil {
		t.Skipf("Model not available at %s: %v", modelPath, err)
	}
	defer reader.Close()

	tok, err := LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		t.Fatalf("Failed to load tokenizer from GGUF: %v", err)
	}

	t.Logf("Tokenizer loaded: vocab_size=%d, bos=%d, eos=%d, unk=%d",
		tok.VocabSize(), tok.bosID, tok.eosID, tok.unkID)

	// Validate each test case
	successCount := 0
	for i, testCase := range referenceData.TestCases {
		testName := fmt.Sprintf("case_%d", i)
		if testCase.Description != "" {
			testName = fmt.Sprintf("case_%d_%s", i, sanitizeTestName(testCase.Description))
		}

		t.Run(testName, func(t *testing.T) {
			// Encode the input text
			actualIDs, err := tok.Encode(testCase.Input)
			if err != nil {
				t.Fatalf("Failed to encode %q: %v", testCase.Input, err)
			}

			// Compare token IDs
			if !slicesEqual(actualIDs, testCase.TokenIDs) {
				// Fail immediately with detailed diff
				if testCase.Description != "" {
					t.Errorf("\n[%s]", testCase.Description)
				}
				t.Errorf("\nTokenization mismatch for input: %q", testCase.Input)
				t.Errorf("\nExpected token IDs (llama.cpp): %v", testCase.TokenIDs)
				t.Errorf("Actual token IDs (our impl):    %v", actualIDs)
				t.Errorf("\nLength: expected=%d, actual=%d", len(testCase.TokenIDs), len(actualIDs))

				// Show detailed position-by-position comparison
				showDetailedDiff(t, tok, testCase.Input, testCase.TokenIDs, actualIDs)

				t.FailNow() // Fail immediately on first mismatch
			}

			successCount++
			if testCase.Description != "" {
				t.Logf("OK [%s]: %q -> %v", testCase.Description, truncateString(testCase.Input, 40), actualIDs)
			} else {
				t.Logf("OK: %q -> %v", truncateString(testCase.Input, 50), actualIDs)
			}
		})
	}

	t.Logf("\nValidation complete: %d/%d test cases passed", successCount, len(referenceData.TestCases))
}

// loadReferenceData loads the reference token IDs from JSON file
func loadReferenceData(path string) (*ReferenceData, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read reference file: %w", err)
	}

	var refData ReferenceData
	if err := json.Unmarshal(data, &refData); err != nil {
		return nil, fmt.Errorf("failed to parse reference JSON: %w", err)
	}

	return &refData, nil
}

// slicesEqual compares two int slices for equality
func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// showDetailedDiff shows a detailed position-by-position comparison of token IDs
func showDetailedDiff(t *testing.T, tok *Tokenizer, input string, expected, actual []int) {
	t.Logf("\nDetailed token comparison:")
	t.Logf("Input text: %q", input)
	t.Logf("Input length: %d bytes, %d runes", len(input), len([]rune(input)))
	t.Logf("")

	maxLen := len(expected)
	if len(actual) > maxLen {
		maxLen = len(actual)
	}

	// Show header
	t.Logf("%-5s %-15s %-40s %-15s %-40s %-8s",
		"Pos", "Expected ID", "Expected Token", "Actual ID", "Actual Token", "Match")
	t.Logf("%s", strings.Repeat("-", 130))

	firstDiff := -1
	for i := 0; i < maxLen; i++ {
		expectedID := -1
		expectedToken := "<missing>"
		if i < len(expected) {
			expectedID = expected[i]
			if expectedID >= 0 && expectedID < tok.VocabSize() {
				expectedToken = truncateString(formatToken(tok.vocab[expectedID]), 38)
			}
		}

		actualID := -1
		actualToken := "<missing>"
		if i < len(actual) {
			actualID = actual[i]
			if actualID >= 0 && actualID < tok.VocabSize() {
				actualToken = truncateString(formatToken(tok.vocab[actualID]), 38)
			}
		}

		match := "OK"
		if expectedID != actualID {
			match = "MISMATCH"
			if firstDiff == -1 {
				firstDiff = i
			}
		}

		// Format IDs
		expectedIDStr := fmt.Sprintf("%d", expectedID)
		if expectedID == -1 {
			expectedIDStr = "-"
		}
		actualIDStr := fmt.Sprintf("%d", actualID)
		if actualID == -1 {
			actualIDStr = "-"
		}

		marker := " "
		if match == "MISMATCH" {
			marker = ">"
		}

		t.Logf("%s%-4d %-15s %-40s %-15s %-40s %-8s",
			marker, i, expectedIDStr, expectedToken, actualIDStr, actualToken, match)
	}

	if firstDiff >= 0 {
		t.Logf("\nFirst difference at position %d", firstDiff)
	}

	// Show normalization and preprocessing steps
	t.Logf("\nTokenization pipeline debug:")
	normalized := tok.normalizer.Normalize(input)
	t.Logf("  1. Original:   %q", input)
	t.Logf("  2. Normalized: %q", normalized)
	withMetaspace := strings.ReplaceAll(normalized, " ", "▁")
	t.Logf("  3. Metaspace:  %q", withMetaspace)
}

// formatToken formats a token string for display, showing special characters
func formatToken(token string) string {
	// Replace common special characters with readable names
	token = strings.ReplaceAll(token, "\n", "\\n")
	token = strings.ReplaceAll(token, "\t", "\\t")
	token = strings.ReplaceAll(token, "\r", "\\r")

	// Show metaspace clearly
	if strings.Contains(token, "▁") {
		return token // Keep metaspace character visible
	}

	return token
}

// truncateString truncates a string to maxLen, adding "..." if truncated
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

// sanitizeTestName converts a description into a valid test name
func sanitizeTestName(desc string) string {
	// Replace spaces and special characters with underscores
	result := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return r
		}
		if r == ' ' || r == ':' || r == '-' {
			return '_'
		}
		return -1 // Remove other characters
	}, desc)

	// Collapse multiple underscores and trim
	result = strings.ReplaceAll(result, "__", "_")
	result = strings.Trim(result, "_")

	// Limit length
	if len(result) > 50 {
		result = result[:50]
	}

	return result
}

// TestReferenceDataFormat validates the structure of the reference data file
// This test ensures the JSON format is correct even before full validation
func TestReferenceDataFormat(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	if _, err := os.Stat(referenceDataPath); os.IsNotExist(err) {
		t.Skipf("Reference data file not found at %s (will be created by Task 2.2)", referenceDataPath)
	}

	// Load and validate reference data
	referenceData, err := loadReferenceData(referenceDataPath)
	if err != nil {
		t.Fatalf("Failed to load reference data: %v", err)
	}

	// Validate structure
	if len(referenceData.TestCases) == 0 {
		t.Error("Reference data contains no test cases")
	}

	t.Logf("Reference data validation:")
	t.Logf("  Total test cases: %d", len(referenceData.TestCases))

	// Check each test case has required fields
	emptyInputCount := 0
	emptyTokensCount := 0
	for i, tc := range referenceData.TestCases {
		if tc.Input == "" {
			emptyInputCount++
			t.Logf("  Warning: Test case %d has empty input", i)
		}
		if len(tc.TokenIDs) == 0 {
			emptyTokensCount++
			t.Logf("  Warning: Test case %d has no token IDs", i)
		}

		// Log first few test cases as examples
		if i < 5 {
			if tc.Description != "" {
				t.Logf("  Case %d [%s]: %q -> %v", i, tc.Description, truncateString(tc.Input, 30), tc.TokenIDs)
			} else {
				t.Logf("  Case %d: %q -> %v", i, truncateString(tc.Input, 40), tc.TokenIDs)
			}
		}
	}

	if emptyInputCount > 0 {
		t.Logf("  Found %d test cases with empty input", emptyInputCount)
	}
	if emptyTokensCount > 0 {
		t.Logf("  Found %d test cases with empty token IDs", emptyTokensCount)
	}

	// Ensure we have at least 20 diverse test cases as per requirements
	if len(referenceData.TestCases) < 20 {
		t.Errorf("Expected at least 20 test cases, got %d", len(referenceData.TestCases))
	}

	t.Logf("\nReference data format is valid")
}

// BenchmarkTokenIDValidation benchmarks the token ID validation process
func BenchmarkTokenIDValidation(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	// Load reference data
	referenceData, err := loadReferenceData(referenceDataPath)
	if err != nil {
		b.Skipf("Reference data not available: %v", err)
	}

	if len(referenceData.TestCases) == 0 {
		b.Skip("No test cases in reference data")
	}

	// Load tokenizer
	reader, err := gguf.Open(modelPath)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer reader.Close()

	tok, err := LoadFromGGUF(reader.GetMetadata)
	if err != nil {
		b.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Use first test case for benchmarking
	testCase := referenceData.TestCases[0]

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(testCase.Input)
	}
}
