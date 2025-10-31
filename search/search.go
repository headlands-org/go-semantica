package search

import (
	"context"
)

// Result captures a nearest-neighbour match.
type Result struct {
	ID       int32
	Distance float32
}

// Builder ingests vectors and produces an immutable index.
type Builder interface {
	// AddVector inserts a pre-computed vector. The vector is expected to match
	// the index dimensionality.
	AddVector(id int32, vec []float32) error

	// Build finalises the builder and returns an Index.
	Build(ctx context.Context) (Index, error)
}

// Index exposes query operations over an immutable vector collection.
type Index interface {
	// Dimension returns the embedding dimensionality.
	Dimension() int

	// SearchVector queries using a pre-computed vector.
	SearchVector(vec []float32, topK int, opts ...SearchOption) ([]Result, error)

	// ForEach iterates over all stored vectors. The provided slice must not be mutated.
	ForEach(fn func(id int32, vec []float32))
}

// Serializer persists and loads indices.
type Serializer interface {
	Serialize(index Index) ([]byte, error)
	Deserialize(data []byte) (Index, error)
}

// SearchOption customises query execution.
type SearchOption interface {
	apply(*Config)
}

// Config describes query-time configuration derived from options.
type Config struct {
	SearchK int
}

// ApplyOptions builds a configuration by applying the provided options.
func ApplyOptions(opts ...SearchOption) Config {
	cfg := Config{}
	for _, opt := range opts {
		opt.apply(&cfg)
	}
	return cfg
}

type searchOptionFunc func(*Config)

func (fn searchOptionFunc) apply(cfg *Config) { fn(cfg) }

// WithSearchK adjusts the upper bound on candidate nodes visited during search.
func WithSearchK(k int) SearchOption {
	return searchOptionFunc(func(cfg *Config) {
		if k > 0 {
			cfg.SearchK = k
		}
	})
}
