// +build integration

package runtime

import (
	"sync"
	"testing"
)

// TestBufferPooling verifies that buffer pooling works correctly
func TestBufferPooling(t *testing.T) {
	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	// Test basic forward pass
	tokenIDs := []int{2, 9259, 1902, 1} // "Hello world"

	// Run multiple times to ensure pool reuse works
	for i := 0; i < 10; i++ {
		embedding, err := model.Forward(tokenIDs)
		if err != nil {
			t.Fatalf("Forward pass %d failed: %v", i, err)
		}

		if len(embedding) != model.config.EmbedDim {
			t.Errorf("Expected embedding dim %d, got %d", model.config.EmbedDim, len(embedding))
		}
	}
}

// TestConcurrentBufferPooling tests thread safety with race detector
// Run with: go test -race -tags=integration ./internal/runtime -run TestConcurrentBufferPooling
func TestConcurrentBufferPooling(t *testing.T) {
	model, err := LoadModel(gemmaModelPath, false)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer model.Close()

	tokenIDs := []int{2, 9259, 1902, 1} // "Hello world"

	const numGoroutines = 4
	const iterationsPerGoroutine = 5

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*iterationsPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for j := 0; j < iterationsPerGoroutine; j++ {
				embedding, err := model.Forward(tokenIDs)
				if err != nil {
					errors <- err
					return
				}

				if len(embedding) != model.config.EmbedDim {
					t.Errorf("Expected embedding dim %d, got %d", model.config.EmbedDim, len(embedding))
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Fatalf("Concurrent forward pass failed: %v", err)
	}
}
