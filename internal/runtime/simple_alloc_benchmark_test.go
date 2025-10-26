// +build integration

package runtime

import (
	"testing"
)

// gemmaModelPath for this test file
const testModelPath = "../../../model/embeddinggemma-300m-Q8_0.gguf"

// BenchmarkAllocations measures memory allocations with pooling
func BenchmarkAllocations(b *testing.B) {
	model, err := LoadModel(testModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	tokenIDs := []int{2, 9259, 1902, 1} // "Hello world"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.Forward(tokenIDs)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// BenchmarkAllocationsLonger measures allocations with longer sequences
func BenchmarkAllocationsLonger(b *testing.B) {
	model, err := LoadModel(testModelPath, false)
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Generate a longer sequence (~20 tokens)
	tokenIDs := []int{
		2, 139, 4773, 3772, 7501, 1809, 708, 671, 3772, 4773,
		1156, 708, 1636, 577, 573, 4038, 578, 6561, 3772, 1,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.Forward(tokenIDs)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}
