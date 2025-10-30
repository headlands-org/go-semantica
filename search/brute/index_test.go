package brute

import (
	"context"
	"math"
	"math/rand"
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

func BenchmarkSearchVectorInt8(b *testing.B) {
	const (
		dimension = 256
		count     = 4096
		topK      = 10
	)

	rng := rand.New(rand.NewSource(42))
	builder := NewBuilder(WithDimension(dimension), WithQuantization(QuantizeInt8))
	for i := 0; i < count; i++ {
		vec := make([]float32, dimension)
		for j := range vec {
			vec[j] = float32(rng.NormFloat64())
		}
		normalize(vec)
		if err := builder.AddVector(int32(i), vec); err != nil {
			b.Fatalf("AddVector: %v", err)
		}
	}
	idxIface, err := builder.Build(context.Background())
	if err != nil {
		b.Fatalf("Build failed: %v", err)
	}
	idx := idxIface.(*Index)

	query := make([]float32, dimension)
	for i := range query {
		query[i] = float32(rng.NormFloat64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.SearchVector(query, topK); err != nil {
			b.Fatalf("SearchVector: %v", err)
		}
	}
}

func TestDotInt8MatchesFloat32(t *testing.T) {
	const (
		dim     = 128
		samples = 100
	)
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < samples; i++ {
		vec := make([]float32, dim)
		query := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(rng.NormFloat64())
			query[j] = float32(rng.NormFloat64())
		}
		normalize(vec)
		normalize(query)

		qVec := quantizeInt8(vec)
		got := dotInt8(qVec, query, dim)
		want := dotFloat32(vec, query)
		if math.Abs(float64(got-want)) > 0.02 {
			t.Fatalf("dot mismatch: got %.6f want %.6f (diff=%.6f)", got, want, got-want)
		}
	}
}

func TestSearchNormalizedMatchesFloat32(t *testing.T) {
	const (
		dim     = 64
		count   = 256
		topK    = 10
		queries = 20
	)
	rng := rand.New(rand.NewSource(456))

	build := func(mode QuantizationMode) *Index {
		builder := NewBuilder(
			WithDimension(dim),
			WithQuantization(mode),
		)
		for i := 0; i < count; i++ {
			vec := make([]float32, dim)
			for j := 0; j < dim; j++ {
				vec[j] = float32(rng.NormFloat64())
			}
			normalize(vec)
			if err := builder.AddVector(int32(i), vec); err != nil {
				t.Fatalf("AddVector: %v", err)
			}
		}
		idx, err := builder.Build(context.Background())
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		return idx.(*Index)
	}

	rng.Seed(456) // rebuild same vectors
	floatIdx := build(QuantizeFloat32)
	rng.Seed(456) // rebuild same vectors for int8 builder
	int8Idx := build(QuantizeInt8)

	for q := 0; q < queries; q++ {
		query := make([]float32, dim)
		for j := range query {
			query[j] = float32(rng.NormFloat64())
		}
		normalize(query)

		floatRes, err := floatIdx.SearchNormalized(query, topK)
		if err != nil {
			t.Fatalf("float SearchNormalized: %v", err)
		}
		int8Res, err := int8Idx.SearchNormalized(query, topK)
		if err != nil {
			t.Fatalf("int8 SearchNormalized: %v", err)
		}
		if len(floatRes) == 0 || len(int8Res) == 0 {
			t.Fatal("empty results")
		}

		if floatRes[0].ID != int8Res[0].ID {
			t.Fatalf("top1 id mismatch: float=%d int8=%d", floatRes[0].ID, int8Res[0].ID)
		}
		diff := math.Abs(float64(floatRes[0].Distance - int8Res[0].Distance))
		if diff > 0.05 {
			t.Fatalf("top1 distance mismatch id=%d diff=%.6f float=%.6f int8=%.6f",
				floatRes[0].ID, diff, floatRes[0].Distance, int8Res[0].Distance)
		}
	}
}
