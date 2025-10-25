// +build integration

package runtime

import (
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func TestDebugWeightShapes(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	reader, err := gguf.Open(gemmaModelPath)
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	defer reader.Close()

	// Check key tensor shapes
	tensors := []string{
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
	}

	t.Log("Weight tensor shapes:")
	for _, name := range tensors {
		if desc, ok := reader.GetTensor(name); ok {
			t.Logf("  %s: %v", name, desc.Shape)
		} else {
			t.Logf("  %s: NOT FOUND", name)
		}
	}
}
