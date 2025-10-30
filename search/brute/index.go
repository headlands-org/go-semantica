package brute

import (
	"errors"
	"fmt"
	"sort"

	"github.com/lth/pure-go-llamas/search"
)

// Index is a quantized brute-force index.
type Index struct {
	dimension    int
	quantization QuantizationMode

	ids     []int32
	idToIdx map[int32]int

	float32Data []float32
	float32Base []byte

	int8Data  []int8
	int8Base  []byte
	int16Data []int16
	int16Base []byte
}

// SearchVector performs a cosine similarity search using the stored vectors.
func (idx *Index) SearchVector(vec []float32, topK int, opts ...search.SearchOption) ([]search.Result, error) {
	if topK <= 0 {
		return nil, fmt.Errorf("brute: topK must be positive")
	}
	if len(vec) != idx.dimension {
		return nil, fmt.Errorf("brute: query dimension mismatch: got %d want %d", len(vec), idx.dimension)
	}

	query := make([]float32, len(vec))
	copy(query, vec)
	normalize(query)

	count := idx.Count()
	scoredResults := make([]struct {
		idx  int
		dist float32
	}, 0, count)

	switch idx.quantization {
	case QuantizeFloat32:
		for i := 0; i < count; i++ {
			dist := dotFloat32(idx.float32Data[i*idx.dimension:(i+1)*idx.dimension], query)
			scoredResults = append(scoredResults, struct {
				idx  int
				dist float32
			}{idx: i, dist: dist})
		}
	case QuantizeInt8:
		for i := 0; i < count; i++ {
			dist := dotInt8(idx.int8Data[i*idx.dimension:(i+1)*idx.dimension], query, idx.dimension)
			scoredResults = append(scoredResults, struct {
				idx  int
				dist float32
			}{idx: i, dist: dist})
		}
	case QuantizeInt16:
		for i := 0; i < count; i++ {
			dist := dotInt16(idx.int16Data[i*idx.dimension:(i+1)*idx.dimension], query, idx.dimension)
			scoredResults = append(scoredResults, struct {
				idx  int
				dist float32
			}{idx: i, dist: dist})
		}
	default:
		return nil, errors.New("brute: unsupported quantization mode")
	}

	sort.Slice(scoredResults, func(i, j int) bool {
		if scoredResults[i].dist == scoredResults[j].dist {
			return idx.ids[scoredResults[i].idx] < idx.ids[scoredResults[j].idx]
		}
		return scoredResults[i].dist > scoredResults[j].dist
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

// Dimension returns the vector dimensionality.
func (idx *Index) Dimension() int { return idx.dimension }

// Count reports the number of stored vectors.
func (idx *Index) Count() int { return len(idx.ids) }

// Vector returns a copy of the stored vector for the given id.
func (idx *Index) Vector(id int32) ([]float32, bool) {
	pos, ok := idx.idToIdx[id]
	if !ok {
		return nil, false
	}
	out := make([]float32, idx.dimension)
	switch idx.quantization {
	case QuantizeFloat32:
		copy(out, idx.float32Data[pos*idx.dimension:(pos+1)*idx.dimension])
	case QuantizeInt8:
		dequantizeInt8(idx.int8Data[pos*idx.dimension:(pos+1)*idx.dimension], out)
	case QuantizeInt16:
		dequantizeInt16(idx.int16Data[pos*idx.dimension:(pos+1)*idx.dimension], out)
	default:
		return nil, false
	}
	return out, true
}

// ForEach iterates over all stored vectors, dequantizing as needed.
func (idx *Index) ForEach(fn func(id int32, vec []float32)) {
	buf := make([]float32, idx.dimension)
	for i, id := range idx.ids {
		switch idx.quantization {
		case QuantizeFloat32:
			copy(buf, idx.float32Data[i*idx.dimension:(i+1)*idx.dimension])
		case QuantizeInt8:
			dequantizeInt8(idx.int8Data[i*idx.dimension:(i+1)*idx.dimension], buf)
		case QuantizeInt16:
			dequantizeInt16(idx.int16Data[i*idx.dimension:(i+1)*idx.dimension], buf)
		}
		fn(id, append([]float32(nil), buf...))
	}
}

func dotFloat32(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
