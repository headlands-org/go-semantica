// generate_token_references.go
//
// Generates reference token IDs using llama.cpp's llama-tokenize tool
// for validating our Go tokenizer implementation.
//
// This is a Go alternative to the shell script version that provides
// more flexibility for programmatic test case generation.
//
// Prerequisites:
//   - llama-tokenize must be installed (from llama.cpp project)
//   - Model file must exist at: model/embeddinggemma-300m-Q8_0.gguf
//
// Usage:
//   go run scripts/generate_token_references.go
//
// Output:
//   testdata/reference_token_ids.json
//
// Build and run:
//   go build -o generate_token_refs scripts/generate_token_references.go
//   ./generate_token_refs

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// TestCase represents a single tokenization test case
type TestCase struct {
	Input       string `json:"input"`
	Description string `json:"description"`
	TokenIDs    []int  `json:"token_ids"`
}

// ReferenceData represents the complete reference dataset
type ReferenceData struct {
	GeneratedBy string     `json:"generated_by"`
	Model       string     `json:"model"`
	TestCases   []TestCase `json:"test_cases"`
}

// tokenize calls llama-tokenize to get token IDs for the given text
func tokenize(modelPath, text string) ([]int, error) {
	cmd := exec.Command("llama-tokenize",
		"-m", modelPath,
		"-p", text,
		"--ids",
		"--log-disable")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("llama-tokenize failed: %w\nOutput: %s", err, string(output))
	}

	// Parse output format: [1, 2, 3]
	outputStr := strings.TrimSpace(string(output))
	if !strings.HasPrefix(outputStr, "[") || !strings.HasSuffix(outputStr, "]") {
		return nil, fmt.Errorf("unexpected output format: %s", outputStr)
	}

	// Remove brackets and parse as JSON array
	var tokenIDs []int
	if err := json.Unmarshal([]byte(outputStr), &tokenIDs); err != nil {
		return nil, fmt.Errorf("failed to parse token IDs: %w", err)
	}

	return tokenIDs, nil
}

// defineTestCases returns all test cases to generate
func defineTestCases() []struct {
	input       string
	description string
} {
	return []struct {
		input       string
		description string
	}{
		// Basic cases
		{"Hello world", "Basic: simple greeting"},
		{"The quick brown fox", "Basic: common phrase"},
		{"Testing tokenization", "Basic: tech phrase"},

		// Empty/minimal
		{"", "Empty: empty string"},
		{"a", "Minimal: single letter"},
		{"ab", "Minimal: two letters"},
		{" ", "Minimal: single space"},

		// Unicode
		{"Hello 👋", "Unicode: emoji wave"},
		{"你好世界", "Unicode: Chinese"},
		{"مرحبا", "Unicode: Arabic"},
		{"こんにちは世界", "Unicode: Japanese"},
		{"Привет мир", "Unicode: Russian"},
		{"γεια σου κόσμε", "Unicode: Greek"},

		// Punctuation
		{"Hello, world!", "Punctuation: comma and exclamation"},
		{"What's this?", "Punctuation: apostrophe and question"},
		{"50% off!", "Punctuation: percent sign"},
		{"test@example.com", "Punctuation: email address"},
		{"It's a test.", "Punctuation: contraction and period"},

		// Whitespace
		{"  spaces  ", "Whitespace: leading and trailing"},
		{"new\nlines", "Whitespace: newline character"},
		{"tab\there", "Whitespace: tab character"},
		{"multiple   spaces", "Whitespace: multiple spaces"},

		// Special characters
		{"#hashtag", "Special: hashtag"},
		{"@mention", "Special: at-mention"},
		{"$100", "Special: dollar sign"},
		{"C++", "Special: programming language"},
		{"foo/bar", "Special: slash"},
		{"key=value", "Special: equals"},

		// Mixed content
		{"你好 world 🌍", "Mixed: Chinese, English, emoji"},
		{"Test #1: 50% done!", "Mixed: numbers, symbols, punctuation"},
		{"Email: user@test.com 📧", "Mixed: email with emoji"},

		// Numbers
		{"123456", "Numbers: digits"},
		{"3.14159", "Numbers: decimal"},
		{"1,000,000", "Numbers: with commas"},

		// Long text
		{"This is a longer sentence with more than thirty words to test how the tokenizer handles longer sequences of text including various punctuation marks and common English words.", "Long: 30+ words"},
		{"The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once, making it useful for testing.", "Long: pangram with explanation"},
	}
}

func main() {
	// Check for llama-tokenize
	if _, err := exec.LookPath("llama-tokenize"); err != nil {
		fmt.Fprintln(os.Stderr, "Error: llama-tokenize not found in PATH")
		fmt.Fprintln(os.Stderr, "Please install llama.cpp: https://github.com/ggerganov/llama.cpp")
		os.Exit(1)
	}

	// Determine paths
	scriptDir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to get script directory: %v\n", err)
		os.Exit(1)
	}
	projectRoot := filepath.Dir(scriptDir)
	modelPath := filepath.Join(projectRoot, "model", "embeddinggemma-300m-Q8_0.gguf")
	outputPath := filepath.Join(projectRoot, "testdata", "reference_token_ids.json")

	// Check model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: Model file not found at %s\n", modelPath)
		fmt.Fprintln(os.Stderr, "Please ensure the model is downloaded or pull from Git LFS:")
		fmt.Fprintln(os.Stderr, `  git lfs pull --include="model/embeddinggemma-300m-Q8_0.gguf"`)
		os.Exit(1)
	}

	fmt.Println("Generating reference token IDs using llama.cpp...")
	fmt.Printf("Model: %s\n\n", modelPath)

	// Generate test cases
	testDefs := defineTestCases()
	testCases := make([]TestCase, 0, len(testDefs))

	for _, testDef := range testDefs {
		tokenIDs, err := tokenize(modelPath, testDef.input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error tokenizing '%s': %v\n", testDef.description, err)
			os.Exit(1)
		}

		testCases = append(testCases, TestCase{
			Input:       testDef.input,
			Description: testDef.description,
			TokenIDs:    tokenIDs,
		})

		fmt.Printf("✓ %s\n", testDef.description)
	}

	// Create reference data
	refData := ReferenceData{
		GeneratedBy: "llama-tokenize (llama.cpp)",
		Model:       "embeddinggemma-300m-Q8_0.gguf",
		TestCases:   testCases,
	}

	// Write JSON output
	jsonData, err := json.MarshalIndent(refData, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling JSON: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(outputPath, jsonData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing output file: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("✓ Successfully generated reference token IDs")
	fmt.Printf("  Output: %s\n", outputPath)
	fmt.Printf("  Test cases: %d\n", len(testCases))
	fmt.Println()
	fmt.Println("You can now use this file in your tokenizer tests to validate")
	fmt.Println("that the Go implementation produces identical token IDs.")
}
