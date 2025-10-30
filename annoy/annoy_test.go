package annoy

import (
	"context"
	"testing"
)

func TestBuilderAndSearchVector(t *testing.T) {
	builder := NewBuilder(
		WithDimension(3),
		WithMetric(Cosine),
		WithNumTrees(10),
		WithMaxLeafSize(1),
		WithSeed(42),
	)

	vecs := map[string][]float32{
		"a": {1, 0, 0},
		"b": {0, 1, 0},
		"c": {0, 0, 1},
	}

	for id, vec := range vecs {
		if err := builder.AddVector(id, vec); err != nil {
			t.Fatalf("AddVector(%s) failed: %v", id, err)
		}
	}

	if err := builder.SetMetadata("b", []byte("payload")); err != nil {
		t.Fatalf("SetMetadata failed: %v", err)
	}

	if err := builder.Build(context.Background()); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	data, err := builder.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}

	index, err := Load(data)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	results, err := index.SearchVector([]float32{1, 0, 0}, 2, WithMetadata())
	if err != nil {
		t.Fatalf("SearchVector failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != "a" {
		t.Fatalf("expected nearest neighbour 'a', got %q", results[0].ID)
	}
	if _, ok := index.Metadata("b"); !ok {
		t.Fatalf("metadata lookup failed")
	}
}
