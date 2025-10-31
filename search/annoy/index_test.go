package annoy

import (
	"context"
	"testing"

	"github.com/headlands-org/go-semantica/search"
)

func TestBuilderSearchAndSerialize(t *testing.T) {
	builder := NewBuilder(
		WithDimension(3),
		WithMetric(Cosine),
		WithNumTrees(8),
		WithMaxLeafSize(1),
		WithSeed(42),
	)

	vectors := map[int32][]float32{
		1: {1, 0, 0},
		2: {0, 1, 0},
		3: {0, 0, 1},
	}
	for id, vec := range vectors {
		if err := builder.AddVector(id, vec); err != nil {
			t.Fatalf("AddVector(%d) failed: %v", id, err)
		}
	}

	idxIface, err := builder.Build(context.Background())
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	index, ok := idxIface.(*Index)
	if !ok {
		t.Fatalf("expected *Index, got %T", idxIface)
	}

	results, err := index.SearchVector([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchVector failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != 1 {
		t.Fatalf("expected nearest neighbour 1, got %d", results[0].ID)
	}

	data, err := index.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}

	reloaded, err := Load(data)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	var seen int
	reloaded.ForEach(func(id int32, vec []float32) {
		seen++
		want := vectors[id]
		for i := range want {
			if want[i] != vec[i] {
				t.Fatalf("vector mismatch for id %d at %d: got %.2f want %.2f", id, i, vec[i], want[i])
			}
		}
	})
	if seen != len(vectors) {
		t.Fatalf("expected %d vectors, saw %d", len(vectors), seen)
	}
}

func BenchmarkSearchVector(b *testing.B) {
	builder := NewBuilder(
		WithDimension(3),
		WithMetric(Cosine),
		WithNumTrees(16),
		WithMaxLeafSize(2),
		WithSeed(123),
	)
	for i := int32(0); i < 512; i++ {
		vec := []float32{float32(i % 7), float32((i + 3) % 7), float32((i + 5) % 7)}
		if err := builder.AddVector(i, vec); err != nil {
			b.Fatalf("AddVector: %v", err)
		}
	}
	idxIface, err := builder.Build(context.Background())
	if err != nil {
		b.Fatalf("Build failed: %v", err)
	}
	index := idxIface.(*Index)
	query := []float32{1, 0, 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := index.SearchVector(query, 10, search.WithSearchK(64)); err != nil {
			b.Fatalf("SearchVector: %v", err)
		}
	}
}
