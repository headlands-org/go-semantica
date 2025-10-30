package annoy

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/lth/pure-go-llamas/search"
)

var errBuilderFinalised = errors.New("annoy: builder already built")

// Builder constructs the Annoy index.
type Builder struct {
	cfg BuilderConfig

	idToIndex map[int32]int
	ids       []int32
	vectors   [][]float32

	trees []*node

	rng *rand.Rand
	mu  sync.Mutex

	built    bool
	progress ProgressFunc
}

// NewBuilder returns a builder with default configuration.
func NewBuilder(opts ...BuilderOption) *Builder {
	cfg := BuilderConfig{
		Metric:      Cosine,
		NumTrees:    20,
		MaxLeafSize: 32,
		Seed:        1,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &Builder{
		cfg:       cfg,
		idToIndex: make(map[int32]int),
		rng:       rand.New(rand.NewSource(cfg.Seed)),
		progress:  cfg.Progress,
	}
}

// AddVector ingests a pre-computed vector with the given identifier.
func (b *Builder) AddVector(id int32, vec []float32) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return errBuilderFinalised
	}
	if _, exists := b.idToIndex[id]; exists {
		return fmt.Errorf("annoy: duplicate id %d", id)
	}

	if b.cfg.Dimension == 0 {
		b.cfg.Dimension = len(vec)
	}
	if len(vec) != b.cfg.Dimension {
		return fmt.Errorf("annoy: vector dimension mismatch: got %d want %d", len(vec), b.cfg.Dimension)
	}

	stored := make([]float32, len(vec))
	copy(stored, vec)
	if b.cfg.Metric.requiresNormalisation() {
		normalise(stored)
	}

	idx := len(b.vectors)
	b.idToIndex[id] = idx
	b.ids = append(b.ids, id)
	b.vectors = append(b.vectors, stored)
	return nil
}

// Build finalises the index.
func (b *Builder) Build(ctx context.Context) (search.Index, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, errBuilderFinalised
	}
	if len(b.vectors) == 0 {
		return nil, errors.New("annoy: no vectors added")
	}
	if b.cfg.Dimension == 0 {
		b.cfg.Dimension = len(b.vectors[0])
	}
	if b.cfg.NumTrees <= 0 {
		b.cfg.NumTrees = 1
	}
	if b.cfg.MaxLeafSize <= 0 {
		b.cfg.MaxLeafSize = 32
	}

	b.trees = make([]*node, b.cfg.NumTrees)
	b.report("build", 0, b.cfg.NumTrees)
	for i := 0; i < b.cfg.NumTrees; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		tree := b.buildTree(i)
		b.trees[i] = tree
		b.report("build", i+1, b.cfg.NumTrees)
	}
	b.built = true

	flat := make([]float32, len(b.vectors)*b.cfg.Dimension)
	for i, vec := range b.vectors {
		copy(flat[i*b.cfg.Dimension:(i+1)*b.cfg.Dimension], vec)
	}

	loaded := &loadedIndex{
		Config: BuilderConfig{
			Dimension:   b.cfg.Dimension,
			Metric:      b.cfg.Metric,
			NumTrees:    b.cfg.NumTrees,
			MaxLeafSize: b.cfg.MaxLeafSize,
		},
		IDs:        append([]int32(nil), b.ids...),
		VectorData: flat,
		Trees:      cloneTrees(b.trees),
	}
	return indexFromLoaded(loaded)
}

func (b *Builder) buildTree(seed int) *node {
	treeRNG := rand.New(rand.NewSource(b.cfg.Seed + int64(seed)*7919))
	indices := make([]int, len(b.vectors))
	for i := range indices {
		indices[i] = i
	}
	return buildNode(indices, b.vectors, b.cfg, treeRNG)
}

func (b *Builder) report(stage string, current, total int) {
	if b.progress != nil {
		b.progress(stage, current, total)
	}
}

func normalise(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	if sum == 0 {
		return
	}
	norm := float32(1.0 / math.Sqrt(sum))
	for i := range vec {
		vec[i] *= norm
	}
}
