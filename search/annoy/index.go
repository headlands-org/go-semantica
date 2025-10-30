package annoy

import (
	"bytes"
	"container/heap"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/lth/pure-go-llamas/search"
)

// Index represents a read-only Annoy index.
type Index struct {
	cfg     BuilderConfig
	ids     []int32
	idToIdx map[int32]int

	vectors []float32
	backing []byte

	trees []*node
}

// Load constructs an Index from the provided binary blob.
func Load(data []byte) (*Index, error) {
	loaded, err := readIndex(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return indexFromLoaded(loaded)
}

// LoadFile reads an index from disk.
func LoadFile(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(data)
}

// Bytes serialises the index into a byte slice.
func (idx *Index) Bytes() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// WriteTo writes the index to w.
func (idx *Index) WriteTo(w io.Writer) (int64, error) {
	serialisable := &serialisableIndex{
		Config:     idx.cfg,
		IDs:        append([]int32(nil), idx.ids...),
		VectorData: append([]float32(nil), idx.vectors...),
		Trees:      cloneTrees(idx.trees),
	}
	return writeIndex(w, serialisable)
}

func indexFromLoaded(loaded *loadedIndex) (*Index, error) {
	idToIdx := make(map[int32]int, len(loaded.IDs))
	for i, id := range loaded.IDs {
		idToIdx[id] = i
	}
	return &Index{
		cfg:     loaded.Config,
		ids:     loaded.IDs,
		idToIdx: idToIdx,
		vectors: loaded.VectorData,
		backing: loaded.VectorBacking,
		trees:   loaded.Trees,
	}, nil
}

// SearchVector queries the index using the configured metric.
func (idx *Index) SearchVector(vec []float32, topK int, opts ...search.SearchOption) ([]search.Result, error) {
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

	cfg := search.ApplyOptions(opts...)
	searchK := cfg.SearchK
	if searchK <= 0 {
		searchK = len(idx.trees) * topK
		if searchK == 0 {
			searchK = len(idx.trees)
		}
	}

	candidates := idx.collectCandidates(query, searchK)
	if len(candidates) < topK {
		total := idx.Count()
		candidates = make([]int, total)
		for i := 0; i < total; i++ {
			candidates[i] = i
		}
	}

	type scored struct {
		idx  int
		dist float32
	}
	scoredResults := make([]scored, 0, len(candidates))
	for _, candidate := range candidates {
		candVec := idx.vectorAt(candidate)
		dist := idx.cfg.Metric.distance(query, candVec)
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

	results := make([]search.Result, topK)
	for i := 0; i < topK; i++ {
		pos := scoredResults[i].idx
		results[i] = search.Result{ID: idx.ids[pos], Distance: scoredResults[i].dist}
	}
	return results, nil
}

func (idx *Index) collectCandidates(query []float32, searchK int) []int {
	seen := make(map[int]struct{})
	pq := make(nodeQueue, len(idx.trees))
	for i, tree := range idx.trees {
		pq[i] = nodeEntry{node: tree, priority: 0}
	}
	heap.Init(&pq)

	visits := 0
	for pq.Len() > 0 && visits < searchK {
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

// Dimension returns the vector dimensionality.
func (idx *Index) Dimension() int { return idx.cfg.Dimension }

// Count returns the number of stored vectors.
func (idx *Index) Count() int { return len(idx.ids) }

// Metric exposes the distance metric for diagnostics.
func (idx *Index) Metric() Metric { return idx.cfg.Metric }

func (idx *Index) vectorAt(pos int) []float32 {
	dim := idx.cfg.Dimension
	start := pos * dim
	end := start + dim
	return idx.vectors[start:end]
}

// ForEach iterates over all stored vectors.
func (idx *Index) ForEach(fn func(id int32, vec []float32)) {
	for i, id := range idx.ids {
		fn(id, idx.vectorAt(i))
	}
}

// Vector copies the vector associated with id.
func (idx *Index) Vector(id int32) ([]float32, bool) {
	pos, ok := idx.idToIdx[id]
	if !ok {
		return nil, false
	}
	vec := idx.vectorAt(pos)
	out := make([]float32, len(vec))
	copy(out, vec)
	return out, true
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
