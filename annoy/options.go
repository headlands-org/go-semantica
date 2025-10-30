package annoy

import (
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

// BuilderOption configures a Builder.
type BuilderOption func(*BuilderConfig)

// BuilderConfig holds configuration shared between builder and final index.
type BuilderConfig struct {
	Dimension   int
	Metric      Metric
	NumTrees    int
	MaxLeafSize int
	Seed        int64

	EmbedRuntime ggufembed.Runtime
	EmbedDim     int
	Progress     ProgressFunc
}

// WithDimension sets the vector dimensionality.
func WithDimension(dim int) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.Dimension = dim
	}
}

// WithMetric selects the distance metric (default Cosine).
func WithMetric(metric Metric) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.Metric = metric
	}
}

// WithNumTrees sets the number of random projection trees to build.
func WithNumTrees(n int) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.NumTrees = n
	}
}

// WithMaxLeafSize sets the maximum number of items stored in a leaf node.
func WithMaxLeafSize(size int) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.MaxLeafSize = size
	}
}

// WithSeed sets the random seed used when constructing the forest.
func WithSeed(seed int64) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.Seed = seed
	}
}

// WithRuntime injects a pre-opened embedding runtime used by AddText/SearchText.
func WithRuntime(rt ggufembed.Runtime) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.EmbedRuntime = rt
	}
}

// WithEmbeddingDim sets the target embedding dimension for Matryoshka truncation.
func WithEmbeddingDim(dim int) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.EmbedDim = dim
	}
}

// ProgressFunc receives updates during long-running operations.
type ProgressFunc func(stage string, current, total int)

// WithProgress registers a callback invoked during embedding/build stages.
func WithProgress(fn ProgressFunc) BuilderOption {
	return func(cfg *BuilderConfig) {
		cfg.Progress = fn
	}
}
