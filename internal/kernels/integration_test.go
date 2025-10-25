// +build integration

package kernels

import (
	"math"
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

const testModelPath = "../../models/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf"

func TestMatMulWithRealQ8_0Weights(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Load real model
	reader, err := gguf.Open(testModelPath)
	if err != nil {
		t.Skipf("Test model not available: %v", err)
	}
	defer reader.Close()

	// Find a Q8_0 weight matrix
	var weightTensor string
	var weightDesc *gguf.TensorDesc

	for _, name := range reader.ListTensors() {
		desc, _ := reader.GetTensor(name)
		if desc.DType == gguf.DTypeQ8_0 && len(desc.Shape) == 2 {
			weightTensor = name
			weightDesc = desc
			break
		}
	}

	if weightTensor == "" {
		t.Skip("No suitable Q8_0 weight tensor found")
	}

	t.Logf("Testing with tensor: %s, shape: %v", weightTensor, weightDesc.Shape)

	// Get weight data
	weightData, err := reader.GetTensorData(weightTensor)
	if err != nil {
		t.Fatalf("Failed to get weight data: %v", err)
	}

	// Matrix dimensions
	M, K, N := 4, weightDesc.Shape[0], weightDesc.Shape[1]

	// Create random activation
	activations := make([]float32, M*K)
	for i := range activations {
		activations[i] = float32(i%10) * 0.1
	}

	// Test Q8_0 matmul
	outputQ8 := make([]float32, M*N)
	err = MatMulQ8_0F32(outputQ8, activations, weightData, M, K, N)
	if err != nil {
		t.Fatalf("Q8_0 matmul failed: %v", err)
	}

	// Verify output is reasonable
	hasNonZero := false
	var minVal, maxVal float32 = math.MaxFloat32, -math.MaxFloat32

	for _, val := range outputQ8 {
		if val != 0 {
			hasNonZero = true
		}
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	if !hasNonZero {
		t.Error("All output values are zero")
	}

	t.Logf("Output statistics:")
	t.Logf("  Min: %.4f", minVal)
	t.Logf("  Max: %.4f", maxVal)
	t.Logf("  Sample: %.4f, %.4f, %.4f", outputQ8[0], outputQ8[len(outputQ8)/2], outputQ8[len(outputQ8)-1])

	// Compare with dequantized F32 matmul
	totalElems := weightDesc.Shape[0] * weightDesc.Shape[1]
	weightF32 := gguf.DequantizeQ8_0(weightData, totalElems)

	outputF32 := make([]float32, M*N)
	MatMulF32(outputF32, activations, weightF32, M, K, N)

	// Calculate difference
	var maxDiff float32
	for i := range outputQ8 {
		diff := float32(math.Abs(float64(outputQ8[i] - outputF32[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("Max difference between Q8_0 and F32 paths: %.6f", maxDiff)

	// They should be very close (Q8_0 matmul should produce same result as dequant-then-matmul)
	if maxDiff > 1e-3 {
		t.Errorf("Q8_0 matmul differs too much from F32 matmul: %.6f", maxDiff)
	}
}

func TestNormalizationWithRealWeights(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := gguf.Open(testModelPath)
	if err != nil {
		t.Skipf("Test model not available: %v", err)
	}
	defer reader.Close()

	// Find a normalization weight tensor
	var normTensor string
	for _, name := range reader.ListTensors() {
		desc, _ := reader.GetTensor(name)
		if desc.DType == gguf.DTypeF32 && len(desc.Shape) == 1 &&
		   (containsString(name, "norm") || containsString(name, "ln")) {
			normTensor = name
			break
		}
	}

	if normTensor == "" {
		t.Skip("No normalization weight tensor found")
	}

	t.Logf("Testing with normalization tensor: %s", normTensor)

	// Get norm weights
	normDesc, _ := reader.GetTensor(normTensor)
	normData, err := reader.GetTensorData(normTensor)
	if err != nil {
		t.Fatalf("Failed to get norm data: %v", err)
	}

	view := gguf.NewTensorView(normDesc, normData)
	normWeights, err := view.AsFloat32()
	if err != nil {
		t.Fatalf("Failed to convert to F32: %v", err)
	}

	dim := len(normWeights)
	t.Logf("  Dimension: %d", dim)

	// Create test input
	input := make([]float32, dim)
	for i := range input {
		input[i] = float32(i%100) * 0.01 - 0.5 // values in [-0.5, 0.49]
	}

	// Test RMSNorm
	output := make([]float32, dim)
	RMSNorm(output, input, normWeights, 1e-6)

	// Verify output
	hasNonZero := false
	var sumSq float32

	for i := range output {
		if output[i] != 0 {
			hasNonZero = true
		}
		sumSq += output[i] * output[i]
	}

	if !hasNonZero {
		t.Error("All normalized values are zero")
	}

	rms := math.Sqrt(float64(sumSq / float32(dim)))
	t.Logf("  Output RMS: %.6f", rms)
	t.Logf("  Sample output: %.4f, %.4f, %.4f", output[0], output[dim/2], output[dim-1])
}

func TestAttentionComponents(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Test attention on small synthetic data
	seqLen, nHeads, headDim := 4, 2, 8
	totalDim := nHeads * headDim

	// Create random Q, K, V
	Q := make([]float32, seqLen*totalDim)
	K := make([]float32, seqLen*totalDim)
	V := make([]float32, seqLen*totalDim)

	for i := range Q {
		Q[i] = float32(i%10) * 0.1
		K[i] = float32((i+5)%10) * 0.1
		V[i] = float32((i+2)%10) * 0.1
	}

	// Apply attention
	output := make([]float32, seqLen*totalDim)
	scratch := make([]float32, seqLen*seqLen+1000) // extra space for scratch

	MultiHeadAttention(output, Q, K, V, 1, seqLen, nHeads, headDim, nil, scratch)

	// Verify output is reasonable
	hasNonZero := false
	for _, val := range output {
		if val != 0 {
			hasNonZero = true
			break
		}
	}

	if !hasNonZero {
		t.Error("Attention output is all zeros")
	}

	t.Logf("Attention output shape: [%d, %d]", seqLen, totalDim)
	t.Logf("Sample values: %.4f, %.4f, %.4f", output[0], output[len(output)/2], output[len(output)-1])

	// Test RoPE
	ropeInput := make([]float32, len(Q))
	copy(ropeInput, Q)

	pos := []int{0, 1, 2, 3}
	ApplyRoPE(ropeInput, seqLen, nHeads, headDim, pos, 10000.0)

	// RoPE should modify the input
	same := true
	for i := range Q {
		if math.Abs(float64(Q[i]-ropeInput[i])) > 1e-6 {
			same = false
			break
		}
	}

	if same {
		t.Error("RoPE did not modify input")
	}

	t.Logf("RoPE successfully applied")
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && s[len(s)-len(substr):] == substr ||
		len(s) > len(substr) && s[:len(substr)] == substr ||
		len(s) > len(substr)*2 && containsInMiddle(s, substr)
}

func containsInMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
