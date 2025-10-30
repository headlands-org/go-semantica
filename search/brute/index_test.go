package brute

import (
	"context"
	"testing"
)

func TestBuilderSearchAndRoundTrip(t *testing.T) {
	builder := NewBuilder(WithDimension(3), WithQuantization(QuantizeInt8))
	input := map[int32][]float32{
		10: {1, 0, 0},
		11: {0, 1, 0},
		12: {0, 0, 1},
	}
	for id, vec := range input {
		if err := builder.AddVector(id, vec); err != nil {
			t.Fatalf("AddVector(%d): %v", id, err)
		}
	}

	idxIface, err := builder.Build(context.Background())
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	idx := idxIface.(*Index)

	results, err := idx.SearchVector([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchVector: %v", err)
	}
	if results[0].ID != 10 {
		t.Fatalf("expected id 10, got %d", results[0].ID)
	}

	ser := Serializer{}
	data, err := ser.Serialize(idx)
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	loadedIface, err := ser.Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialize: %v", err)
	}
	loaded := loadedIface.(*Index)

	if loaded.Dimension() != idx.Dimension() {
		t.Fatalf("dimension mismatch: got %d want %d", loaded.Dimension(), idx.Dimension())
	}
	if loaded.Count() != len(input) {
		t.Fatalf("count mismatch: got %d want %d", loaded.Count(), len(input))
	}

	for id := range input {
		vec, ok := loaded.Vector(id)
		if !ok {
			t.Fatalf("missing vector for id %d", id)
		}
		if len(vec) != idx.Dimension() {
			t.Fatalf("vector length mismatch for id %d", id)
		}
	}
}
