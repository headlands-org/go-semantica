package annoy

import (
	"container/heap"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/lth/pure-go-llamas/model"
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

// Index represents a read-only Annoy index.
type Index struct {
	cfg      BuilderConfig
	ids      []string
	idToIdx  map[string]int
	vectors  [][]float32
	metadata [][]byte
	trees    []*node

	runtimeOnce sync.Once
	runtime     ggufembed.Runtime
	runtimeErr  error
}

// Load constructs an Index from the provided binary blob.
func Load(data []byte) (*Index, error) {
	loaded, err := readIndex(data)
	if err != nil {
		return nil, err
	}
	return indexFromLoaded(loaded)
}

// LoadReader consumes all bytes from r and loads the index.
func LoadReader(r io.Reader) (*Index, error) {
	buf, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return Load(buf)
}

// LoadFile reads an index from disk.
func LoadFile(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(data)
}

func indexFromLoaded(loaded *loadedIndex) (*Index, error) {
	idToIdx := make(map[string]int, len(loaded.IDs))
	for i, id := range loaded.IDs {
		idToIdx[id] = i
	}
	cfg := loaded.Config
	if cfg.Dimension == 0 && len(loaded.Vectors) > 0 {
		cfg.Dimension = len(loaded.Vectors[0])
	}

	return &Index{
		cfg:      cfg,
		ids:      loaded.IDs,
		idToIdx:  idToIdx,
		vectors:  loaded.Vectors,
		metadata: loaded.Metadata,
		trees:    loaded.Trees,
	}, nil
}

// Result contains the outcome of a search.
type Result struct {
	ID       string
	Distance float32
	Metadata []byte
}

// SearchOption customises query behaviour.
type SearchOption func(*searchConfig)

type searchConfig struct {
	searchK     int
	includeMeta bool
}

// WithSearchK sets an upper bound on the number of leaf candidates visited per tree.
func WithSearchK(k int) SearchOption {
	return func(cfg *searchConfig) {
		if k > 0 {
			cfg.searchK = k
		}
	}
}

// WithMetadata requests metadata payloads to be copied into the results.
func WithMetadata() SearchOption {
	return func(cfg *searchConfig) { cfg.includeMeta = true }
}

// SearchVector queries the index with a pre-computed vector.
func (idx *Index) SearchVector(vec []float32, topK int, opts ...SearchOption) ([]Result, error) {
	if topK <= 0 {
		return nil, fmt.Errorf("annoy: topK must be positive")
	}
	if len(vec) != idx.cfg.Dimension {
		return nil, fmt.Errorf("annoy: query vector dimension mismatch: got %d want %d", len(vec), idx.cfg.Dimension)
	}

	query := make([]float32, len(vec))
	copy(query, vec)
	if idx.cfg.Metric.requiresNormalisation() {
		normalise(query)
	}

	cfg := searchConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.searchK <= 0 {
		cfg.searchK = len(idx.trees) * topK
		if cfg.searchK == 0 {
			cfg.searchK = len(idx.trees)
		}
	}

	candidates := idx.collectCandidates(query, cfg, topK)
	if len(candidates) < topK {
		// Fall back to brute-force search to guarantee enough results.
		candidates = make([]int, len(idx.vectors))
		for i := range idx.vectors {
			candidates[i] = i
		}
	}

	type scored struct {
		idx  int
		dist float32
	}
	scoredResults := make([]scored, 0, len(candidates))
	for _, candidate := range candidates {
		vec := idx.vectors[candidate]
		dist := idx.cfg.Metric.distance(query, vec)
		scoredResults = append(scoredResults, scored{idx: candidate, dist: dist})
	}

	sort.Slice(scoredResults, func(i, j int) bool {
		if scoredResults[i].dist == scoredResults[j].dist {
			return idx.ids[scoredResults[i].idx] < idx.ids[scoredResults[j].idx]
		}
		return scoredResults[i].dist < scoredResults[j].dist
	})

	if topK > len(scoredResults) {
		topK = len(scoredResults)
	}

	results := make([]Result, topK)
	for i := 0; i < topK; i++ {
		idxVal := scoredResults[i].idx
		res := Result{ID: idx.ids[idxVal], Distance: scoredResults[i].dist}
		if cfg.includeMeta && len(idx.metadata[idxVal]) > 0 {
			meta := make([]byte, len(idx.metadata[idxVal]))
			copy(meta, idx.metadata[idxVal])
			res.Metadata = meta
		}
		results[i] = res
	}
	return results, nil
}

type nodeEntry struct {
	node     *node
	priority float32
}

type nodeQueue []nodeEntry

func (h nodeQueue) Len() int           { return len(h) }
func (h nodeQueue) Less(i, j int) bool { return h[i].priority < h[j].priority }
func (h nodeQueue) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *nodeQueue) Push(x interface{}) {
	*h = append(*h, x.(nodeEntry))
}

func (h *nodeQueue) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

func (idx *Index) collectCandidates(query []float32, cfg searchConfig, topK int) []int {
	seen := make(map[int]struct{})
	pq := make(nodeQueue, len(idx.trees))
	for i, tree := range idx.trees {
		pq[i] = nodeEntry{node: tree, priority: 0}
	}
	heap.Init(&pq)

	visits := 0
	maxVisits := cfg.searchK
	for pq.Len() > 0 && visits < maxVisits {
		entry := heap.Pop(&pq).(nodeEntry)
		n := entry.node
		if n == nil {
			continue
		}
		if n.leaf {
			visits++
			for _, idx := range n.indices {
				seen[idx] = struct{}{}
			}
			continue
		}
		score := dot(n.hyperplane, query)
		diff := score - n.threshold
		near, far := n.left, n.right
		if diff > 0 {
			near, far = n.right, n.left
		}
		priority := float32(math.Abs(float64(diff)))
		heap.Push(&pq, nodeEntry{node: near, priority: priority})
		heap.Push(&pq, nodeEntry{node: far, priority: priority + 1e-6})
	}

	out := make([]int, 0, len(seen))
	for cand := range seen {
		out = append(out, cand)
	}
	return out
}

// SearchText embeds the query string using the embedded runtime and performs a search.
func (idx *Index) SearchText(ctx context.Context, text string, topK int, opts ...SearchOption) ([]Result, error) {
	vec, err := idx.EmbedQuery(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("annoy: embed query: %w", err)
	}
	return idx.SearchVector(vec, topK, opts...)
}

// EmbedQuery embeds query text using the index's runtime.
func (idx *Index) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	rt, err := idx.getRuntime()
	if err != nil {
		return nil, err
	}
	return rt.EmbedSingleInput(ctx, ggufembed.EmbedInput{
		Task:    ggufembed.TaskSearchQuery,
		Content: text,
		Dim:     idx.cfg.Dimension,
	})
}

func (idx *Index) getRuntime() (ggufembed.Runtime, error) {
	idx.runtimeOnce.Do(func() {
		rt, err := model.Open()
		if err != nil {
			idx.runtimeErr = fmt.Errorf("annoy: open embedded runtime: %w", err)
			return
		}
		idx.runtime = rt
	})
	return idx.runtime, idx.runtimeErr
}

// ForEachVector iterates over all stored vectors. The provided slice must not be mutated.
func (idx *Index) ForEachVector(fn func(id string, vec []float32)) {
	for i, id := range idx.ids {
		fn(id, idx.vectors[i])
	}
}

// Dimension returns the embedding dimension of the index.
func (idx *Index) Dimension() int { return idx.cfg.Dimension }

// Metric returns the distance metric used by the index.
func (idx *Index) Metric() Metric { return idx.cfg.Metric }

// GetVector returns a copy of the stored vector for a given identifier.
func (idx *Index) GetVector(id string) ([]float32, bool) {
	index, ok := idx.idToIdx[id]
	if !ok {
		return nil, false
	}
	vec := make([]float32, len(idx.vectors[index]))
	copy(vec, idx.vectors[index])
	return vec, true
}

// Metadata returns the metadata payload for the given identifier.
func (idx *Index) Metadata(id string) ([]byte, bool) {
	index, ok := idx.idToIdx[id]
	if !ok {
		return nil, false
	}
	data := idx.metadata[index]
	if data == nil {
		return nil, true
	}
	out := make([]byte, len(data))
	copy(out, data)
	return out, true
}

// Items returns all stored identifiers.
func (idx *Index) Items() []string {
	out := make([]string, len(idx.ids))
	copy(out, idx.ids)
	return out
}
