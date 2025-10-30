package kernels

import (
	"math"
	"testing"
)

// TestMultiHeadAttentionParallel verifies that parallel head processing produces identical results
func TestMultiHeadAttentionParallel(t *testing.T) {
	testCases := []struct {
		name     string
		seqLen   int
		nHeads   int
		headDim  int
		useChunk bool
	}{
		{"SingleQuery_8Heads", 1, 8, 64, false},
		{"SingleQuery_16Heads", 1, 16, 64, false},
		{"SingleQuery_32Heads", 1, 32, 64, false},
		{"MultiQuery_8Heads", 8, 8, 64, false},
		{"MultiQuery_16Heads", 16, 16, 64, false},
		{"MultiQuery_8Heads_Chunked", 128, 8, 64, true},
		{"MultiQuery_16Heads_Chunked", 128, 16, 64, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			totalDim := tc.nHeads * tc.headDim

			// Create deterministic test data
			Q := make([]float32, tc.seqLen*totalDim)
			K := make([]float32, tc.seqLen*totalDim)
			V := make([]float32, tc.seqLen*totalDim)

			for i := range Q {
				Q[i] = float32(i%13) * 0.1
				K[i] = float32((i+7)%13) * 0.1
				V[i] = float32((i+3)%13) * 0.1
			}

			// Serial execution (no runTasks)
			outputSerial := make([]float32, tc.seqLen*totalDim)
			scratchSerial := make([]float32, tc.seqLen*tc.seqLen*tc.nHeads)

			if tc.useChunk {
				chunkSize := 64
				MultiHeadAttentionChunked(outputSerial, Q, K, V, 1, tc.seqLen, tc.nHeads, tc.headDim, nil, 1.0, scratchSerial, chunkSize, nil, 8)
			} else {
				MultiHeadAttentionWithScale(outputSerial, Q, K, V, 1, tc.seqLen, tc.nHeads, tc.headDim, nil, 1.0, scratchSerial)
			}

			// Parallel execution with mock task runner
			outputParallel := make([]float32, tc.seqLen*totalDim)
			scratchParallel := make([]float32, tc.seqLen*tc.seqLen*tc.nHeads)

			mockRunner := func(tasks ...func()) {
				// Execute tasks in order (simulating parallel execution)
				for _, task := range tasks {
					task()
				}
			}

			if tc.useChunk {
				chunkSize := 64
				MultiHeadAttentionChunked(outputParallel, Q, K, V, 1, tc.seqLen, tc.nHeads, tc.headDim, nil, 1.0, scratchParallel, chunkSize, mockRunner, 8)
			} else {
				MultiHeadAttentionChunked(outputParallel, Q, K, V, 1, tc.seqLen, tc.nHeads, tc.headDim, nil, 1.0, scratchParallel, 0, mockRunner, 8)
			}

			// Compare outputs
			maxDiff := float32(0)
			for i := range outputSerial {
				diff := float32(math.Abs(float64(outputSerial[i] - outputParallel[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Allow small numerical differences
			if maxDiff > 1e-5 {
				t.Errorf("Outputs differ: max difference = %e", maxDiff)

				// Print first few differences for debugging
				debugCount := 10
				if len(outputSerial) < debugCount {
					debugCount = len(outputSerial)
				}
				for i := 0; i < debugCount; i++ {
					if math.Abs(float64(outputSerial[i]-outputParallel[i])) > 1e-6 {
						t.Logf("  [%d] serial=%f parallel=%f diff=%e", i, outputSerial[i], outputParallel[i],
							math.Abs(float64(outputSerial[i]-outputParallel[i])))
					}
				}
			} else {
				t.Logf("Outputs match (max diff: %e)", maxDiff)
			}

			// Verify outputs are non-trivial
			hasNonZero := false
			for _, val := range outputSerial {
				if val != 0 {
					hasNonZero = true
					break
				}
			}
			if !hasNonZero {
				t.Error("Output is all zeros")
			}
		})
	}
}

// TestMultiHeadAttentionHeadGrouping verifies head grouping strategy
func TestMultiHeadAttentionHeadGrouping(t *testing.T) {
	testCases := []struct {
		name              string
		seqLen            int
		nHeads            int
		expectedTaskCount int // Based on headsPerTask logic
	}{
		{"SingleQuery_8Heads", 1, 8, 0},    // No parallelization (seqLen=1, nHeads < 16)
		{"SingleQuery_16Heads", 1, 16, 16}, // 1 head per task for seqLen=1
		{"SingleQuery_32Heads", 1, 32, 32}, // 1 head per task for seqLen=1
		{"MultiQuery_8Heads", 8, 8, 4},     // 2 heads per task (nHeads >= 8)
		{"MultiQuery_16Heads", 8, 16, 4},   // 4 heads per task (nHeads >= 16)
		{"MultiQuery_32Heads", 8, 32, 8},   // 4 heads per task (nHeads >= 16)
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			totalDim := tc.nHeads * 64
			Q := make([]float32, tc.seqLen*totalDim)
			K := make([]float32, tc.seqLen*totalDim)
			V := make([]float32, tc.seqLen*totalDim)
			output := make([]float32, tc.seqLen*totalDim)
			scratch := make([]float32, tc.seqLen*tc.seqLen*tc.nHeads)

			taskCount := 0
			mockRunner := func(tasks ...func()) {
				taskCount = len(tasks)
				for _, task := range tasks {
					task()
				}
			}

			MultiHeadAttentionChunked(output, Q, K, V, 1, tc.seqLen, tc.nHeads, 64, nil, 1.0, scratch, 0, mockRunner, 8)

			if taskCount != tc.expectedTaskCount {
				t.Errorf("Expected %d tasks, got %d", tc.expectedTaskCount, taskCount)
			} else {
				t.Logf("Correctly created %d parallel tasks", taskCount)
			}
		})
	}
}
