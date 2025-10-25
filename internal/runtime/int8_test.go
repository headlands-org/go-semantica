// +build integration

package runtime

import (
	"math"
	"testing"
)

func TestINT8Accuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping INT8 test in short mode")
	}

	// Load model
	model, err := LoadModel(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	text := "Hello world"
	t.Logf("Testing text: %q", text)

	// Tokenize
	tokenIDs, err := model.Tokenizer().Encode(text)
	if err != nil {
		t.Fatalf("Tokenization failed: %v", err)
	}
	t.Logf("Token IDs: %v", tokenIDs)

	// FP32 inference (baseline)
	t.Log("\n=== FP32 Inference ===")
	embFP32, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("FP32 forward failed: %v", err)
	}
	t.Logf("FP32 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
		embFP32[0], embFP32[1], embFP32[2], embFP32[3], embFP32[4],
		embFP32[5], embFP32[6], embFP32[7], embFP32[8], embFP32[9])

	// INT8 inference
	t.Log("\n=== INT8 Inference ===")
	embINT8, err := model.ForwardINT8(tokenIDs)
	if err != nil {
		t.Fatalf("INT8 forward failed: %v", err)
	}
	t.Logf("INT8 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
		embINT8[0], embINT8[1], embINT8[2], embINT8[3], embINT8[4],
		embINT8[5], embINT8[6], embINT8[7], embINT8[8], embINT8[9])

	// Compare embeddings
	if len(embFP32) != len(embINT8) {
		t.Fatalf("Dimension mismatch: FP32=%d, INT8=%d", len(embFP32), len(embINT8))
	}

	// Compute cosine similarity
	cosineSim := cosineSimilarity(embFP32, embINT8)
	t.Logf("\nCosine similarity (INT8 vs FP32): %.6f", cosineSim)

	// Compute numerical differences
	maxDiff := float32(0)
	maxDiffIdx := 0
	sumDiff := float32(0)
	sumSqDiff := float32(0)

	for i := range embFP32 {
		diff := float32(math.Abs(float64(embFP32[i] - embINT8[i])))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
		sumDiff += diff
		sumSqDiff += diff * diff
	}

	avgDiff := sumDiff / float32(len(embFP32))
	rmsDiff := float32(math.Sqrt(float64(sumSqDiff / float32(len(embFP32)))))

	t.Logf("\nNumerical comparison:")
	t.Logf("  Max difference:  %.6f at index %d", maxDiff, maxDiffIdx)
	t.Logf("    FP32 value:    %.6f", embFP32[maxDiffIdx])
	t.Logf("    INT8 value:    %.6f", embINT8[maxDiffIdx])
	t.Logf("  Avg difference:  %.6f", avgDiff)
	t.Logf("  RMS difference:  %.6f", rmsDiff)

	// Check accuracy threshold
	// INT8 quantization will have some loss, but should maintain >0.95 similarity
	threshold := float32(0.95)
	if cosineSim < threshold {
		t.Errorf("❌ INT8 accuracy too low: %.6f < %.2f", cosineSim, threshold)
	} else {
		t.Logf("\n✅ INT8 accuracy acceptable: %.6f >= %.2f", cosineSim, threshold)
	}
}

func BenchmarkForwardINT8(b *testing.B) {
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
		_, _ = model.ForwardINT8(tokenIDs)
	}
}
