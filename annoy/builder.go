package annoy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"github.com/lth/pure-go-llamas/model"
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

var (
	errBuilderFinalised = errors.New("annoy: builder already built")
	errUnknownID        = errors.New("annoy: unknown id")
)

// Builder constructs a forest of random projection trees and serialises the
// resulting index.
type Builder struct {
	cfg BuilderConfig

	idToIndex map[string]int
	ids       []string
	vectors   [][]float32
	metadata  [][]byte

	trees []*node

	rng *rand.Rand
	mu  sync.Mutex

	defaultRuntimeOnce sync.Once
	defaultRuntime     ggufembed.Runtime
	runtimeErr         error

	built bool

	progress ProgressFunc
}

// Document represents a text payload to embed and index.
type Document struct {
	ID       string
	Title    string
	Text     string
	Metadata []byte
}

// NewBuilder creates a new builder with the provided options.
func NewBuilder(opts ...BuilderOption) *Builder {
	cfg := BuilderConfig{
		Metric:      Cosine,
		NumTrees:    20,
		MaxLeafSize: 32,
		Seed:        1,
		EmbedDim:    ggufembed.DefaultEmbedDim,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &Builder{
		cfg:                cfg,
		idToIndex:          make(map[string]int),
		rng:                rand.New(rand.NewSource(cfg.Seed)),
		metadata:           make([][]byte, 0),
		vectors:            make([][]float32, 0),
		ids:                make([]string, 0),
		progress:           cfg.Progress,
		defaultRuntimeOnce: sync.Once{},
	}
}

// AddVector inserts a pre-computed vector. Vectors are implicitly normalised
// when using the cosine metric.
func (b *Builder) AddVector(id string, vec []float32) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return errBuilderFinalised
	}
	if _, exists := b.idToIndex[id]; exists {
		return fmt.Errorf("annoy: duplicate id %q", id)
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
	b.metadata = append(b.metadata, nil)
	return nil
}

// AddText embeds the provided text using the configured runtime (or the default
// embedded Gemma model) and adds it to the index.
func (b *Builder) AddText(ctx context.Context, id, text string, meta []byte) error {
	return b.AddBatchTexts(ctx, []Document{{ID: id, Text: text, Metadata: meta}}, 1)
}

// AddBatchTexts embeds multiple documents in batches and inserts them.
func (b *Builder) AddBatchTexts(ctx context.Context, docs []Document, batchSize int) error {
	rt, err := b.getRuntime()
	if err != nil {
		return err
	}
	if batchSize <= 0 {
		batchSize = 128
	}
	total := len(docs)
	b.report("embed", 0, total)
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		batch := docs[start:end]
		inputs := make([]ggufembed.EmbedInput, len(batch))
		for i, doc := range batch {
			inputs[i] = ggufembed.EmbedInput{
				Task:    ggufembed.TaskSearchDocument,
				Title:   doc.Title,
				Content: doc.Text,
				Dim:     b.cfg.EmbedDim,
			}
		}
		vectors, err := rt.EmbedInputs(ctx, inputs)
		if err != nil {
			return fmt.Errorf("annoy: batch embed: %w", err)
		}
		for i, vec := range vectors {
			doc := batch[i]
			if err := b.AddVector(doc.ID, vec); err != nil {
				return err
			}
			if len(doc.Metadata) > 0 {
				if err := b.SetMetadata(doc.ID, doc.Metadata); err != nil {
					return err
				}
			}
		}
		b.report("embed", end, total)
	}
	return nil
}

// SetMetadata associates arbitrary binary payload with an item.
func (b *Builder) SetMetadata(id string, payload []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	idx, ok := b.idToIndex[id]
	if !ok {
		return errUnknownID
	}
	buf := make([]byte, len(payload))
	copy(buf, payload)
	b.metadata[idx] = buf
	return nil
}

// Size returns the number of items added so far.
func (b *Builder) Size() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.vectors)
}

// Build constructs the forest. After Build, no more vectors may be added.
func (b *Builder) Build(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return errBuilderFinalised
	}
	if len(b.vectors) == 0 {
		return errors.New("annoy: no vectors added")
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
			return ctx.Err()
		default:
		}
		tree := b.buildTree(i)
		b.trees[i] = tree
		b.report("build", i+1, b.cfg.NumTrees)
	}
	b.built = true
	return nil
}

// Bytes serialises the built index into a compact binary blob.
func (b *Builder) Bytes() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := b.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// WriteTo serialises the index into w. Build must have been called.
func (b *Builder) WriteTo(w io.Writer) (int64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.built {
		return 0, errors.New("annoy: Build must be called before serialisation")
	}
	return writeIndex(w, &serialisableIndex{
		Config:   b.cfg,
		IDs:      b.ids,
		Vectors:  b.vectors,
		Metadata: b.metadata,
		Trees:    b.trees,
	})
}

// WriteEmbeddings writes the embedding matrix as JSON records after Build.
func (b *Builder) WriteEmbeddings(w io.Writer) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.built {
		return errors.New("annoy: Build must be called before writing embeddings")
	}
	enc := json.NewEncoder(w)
	type record struct {
		ID     string    `json:"id"`
		Vector []float32 `json:"vector"`
	}
	for i, id := range b.ids {
		if err := enc.Encode(record{ID: id, Vector: b.vectors[i]}); err != nil {
			return err
		}
	}
	return nil
}

func (b *Builder) buildTree(seed int) *node {
	treeRNG := rand.New(rand.NewSource(b.cfg.Seed + int64(seed)*7919))
	indices := make([]int, len(b.vectors))
	for i := range indices {
		indices[i] = i
	}
	return buildNode(indices, b.vectors, b.cfg, treeRNG)
}

func (b *Builder) getRuntime() (ggufembed.Runtime, error) {
	if b.cfg.EmbedRuntime != nil {
		return b.cfg.EmbedRuntime, nil
	}
	b.defaultRuntimeOnce.Do(func() {
		rt, err := model.Open()
		if err != nil {
			b.runtimeErr = fmt.Errorf("annoy: open embedded runtime: %w", err)
			return
		}
		b.defaultRuntime = rt
	})
	return b.defaultRuntime, b.runtimeErr
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

// serialisableIndex aggregates data required for persistence.
type serialisableIndex struct {
	Config   BuilderConfig
	IDs      []string
	Vectors  [][]float32
	Metadata [][]byte
	Trees    []*node
}
