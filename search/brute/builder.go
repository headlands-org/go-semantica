package brute

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lth/pure-go-llamas/search"
)

var errBuilderFinalised = errors.New("brute: builder already built")

// Builder constructs a quantized brute-force index.
type Builder struct {
	dimension    int
	quantization QuantizationMode

	ids     []int32
	vectors [][]float32
	idSet   map[int32]struct{}

	built bool
}

// BuilderOption configures the builder.
type BuilderOption func(*Builder)

// WithDimension sets the vector dimension up front.
func WithDimension(dim int) BuilderOption {
	return func(b *Builder) { b.dimension = dim }
}

// WithQuantization selects the storage precision.
func WithQuantization(mode QuantizationMode) BuilderOption {
	return func(b *Builder) { b.quantization = mode }
}

// NewBuilder returns a Builder with sensible defaults.
func NewBuilder(opts ...BuilderOption) *Builder {
	b := &Builder{
		quantization: QuantizeInt8,
		idSet:        make(map[int32]struct{}),
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// AddVector inserts a pre-computed vector.
func (b *Builder) AddVector(id int32, vec []float32) error {
	if b.built {
		return errBuilderFinalised
	}
	if _, exists := b.idSet[id]; exists {
		return fmt.Errorf("brute: duplicate id %d", id)
	}
	if b.dimension == 0 {
		b.dimension = len(vec)
	}
	if len(vec) != b.dimension {
		return fmt.Errorf("brute: vector dimension mismatch: got %d want %d", len(vec), b.dimension)
	}

	stored := make([]float32, len(vec))
	copy(stored, vec)
	normalize(stored)

	b.ids = append(b.ids, id)
	b.vectors = append(b.vectors, stored)
	b.idSet[id] = struct{}{}
	return nil
}

// Build materialises the read-only index.
func (b *Builder) Build(ctx context.Context) (search.Index, error) {
	if b.built {
		return nil, errBuilderFinalised
	}
	if len(b.vectors) == 0 {
		return nil, errors.New("brute: no vectors added")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	idx := &Index{
		dimension:    b.dimension,
		quantization: b.quantization,
		ids:          append([]int32(nil), b.ids...),
		idToIdx:      make(map[int32]int, len(b.ids)),
	}
	for i, id := range idx.ids {
		idx.idToIdx[id] = i
	}

	switch b.quantization {
	case QuantizeFloat32:
		data := make([]float32, len(b.vectors)*b.dimension)
		for i, vec := range b.vectors {
			copy(data[i*b.dimension:(i+1)*b.dimension], vec)
		}
		idx.float32Data = data
	case QuantizeInt8:
		data := make([]int8, len(b.vectors)*b.dimension)
		for i, vec := range b.vectors {
			q := quantizeInt8(vec)
			copy(data[i*b.dimension:(i+1)*b.dimension], q)
		}
		idx.int8Data = data
	case QuantizeInt16:
		data := make([]int16, len(b.vectors)*b.dimension)
		for i, vec := range b.vectors {
			q := quantizeInt16(vec)
			copy(data[i*b.dimension:(i+1)*b.dimension], q)
		}
		idx.int16Data = data
	default:
		return nil, fmt.Errorf("brute: unsupported quantization %d", b.quantization)
	}

	b.built = true
	return idx, nil
}

func normalize(vec []float32) {
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
