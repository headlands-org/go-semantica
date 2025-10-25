// +build integration

package runtime

import (
	"testing"
)

func BenchmarkTokenization(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	text := "This is a test sentence for benchmarking tokenization speed."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.tokenizer.Encode(text)
	}
}

func BenchmarkForward(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	tokenIDs := []int{2, 9259, 1902, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Forward(tokenIDs)
	}
}

func BenchmarkEndToEnd(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	text := "Hello world"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenIDs, _ := model.tokenizer.Encode(text)
		_, _ = model.Forward(tokenIDs)
	}
}
