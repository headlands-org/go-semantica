package annoy

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unsafe"

	"github.com/headlands-org/go-semantica/search"
)

var fileMagic = [4]byte{'A', 'N', 'N', 'G'}

const fileVersion uint16 = 1

type fileHeader struct {
	Magic       [4]byte
	Version     uint16
	Metric      uint16
	Dimension   uint16
	NumTrees    uint16
	MaxLeaf     uint16
	Reserved    uint16
	VectorCount uint32
}

type countingWriter struct {
	w io.Writer
	n int64
}

type serialisableIndex struct {
	Config     BuilderConfig
	IDs        []int32
	VectorData []float32
	Trees      []*node
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

func writeIndex(w io.Writer, idx *serialisableIndex) (int64, error) {
	cw := &countingWriter{w: w}
	dim := idx.Config.Dimension
	if dim == 0 {
		if len(idx.IDs) == 0 {
			return 0, errors.New("annoy: zero-dimension index with no vectors")
		}
		if len(idx.VectorData) == 0 {
			return 0, errors.New("annoy: missing vector payload")
		}
		dim = len(idx.VectorData) / len(idx.IDs)
		idx.Config.Dimension = dim
	}
	header := fileHeader{
		Magic:       fileMagic,
		Version:     fileVersion,
		Metric:      uint16(idx.Config.Metric),
		Dimension:   uint16(dim),
		NumTrees:    uint16(len(idx.Trees)),
		MaxLeaf:     uint16(idx.Config.MaxLeafSize),
		VectorCount: uint32(len(idx.IDs)),
	}

	if err := binary.Write(cw, binary.LittleEndian, header); err != nil {
		return cw.n, err
	}

	if len(idx.IDs) > 0 {
		if err := binary.Write(cw, binary.LittleEndian, idx.IDs); err != nil {
			return cw.n, fmt.Errorf("annoy: write ids: %w", err)
		}
	}
	if len(idx.VectorData) > 0 {
		if err := binary.Write(cw, binary.LittleEndian, idx.VectorData); err != nil {
			return cw.n, fmt.Errorf("annoy: write vectors: %w", err)
		}
	}

	for _, tree := range idx.Trees {
		if err := writeTree(cw, tree); err != nil {
			return cw.n, err
		}
	}
	return cw.n, nil
}

const (
	nodeLeaf     = byte(1)
	nodeInternal = byte(0)
)

func writeTree(w io.Writer, n *node) error {
	if n.leaf {
		if _, err := w.Write([]byte{nodeLeaf}); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(len(n.indices))); err != nil {
			return err
		}
		for _, idx := range n.indices {
			if err := binary.Write(w, binary.LittleEndian, uint32(idx)); err != nil {
				return err
			}
		}
		return nil
	}

	if _, err := w.Write([]byte{nodeInternal}); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, n.hyperplane); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, n.threshold); err != nil {
		return err
	}
	if err := writeTree(w, n.left); err != nil {
		return err
	}
	return writeTree(w, n.right)
}

type loadedIndex struct {
	Config        BuilderConfig
	IDs           []int32
	VectorData    []float32
	VectorBacking []byte
	Trees         []*node
}

// Serializer implements search.Serializer for Annoy indices.
type Serializer struct{}

// Serialize writes the index to bytes.
func (Serializer) Serialize(idx search.Index) ([]byte, error) {
	ann, ok := idx.(*Index)
	if !ok {
		return nil, fmt.Errorf("annoy: serializer expects *Index, got %T", idx)
	}
	return ann.Bytes()
}

// Deserialize reconstructs an Annoy index from bytes.
func (Serializer) Deserialize(data []byte) (search.Index, error) {
	return Load(data)
}

func readIndex(r io.Reader) (*loadedIndex, error) {
	var header fileHeader
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return nil, err
	}
	if header.Magic != fileMagic {
		return nil, errors.New("annoy: invalid index magic")
	}
	if header.Version != fileVersion {
		return nil, fmt.Errorf("annoy: unsupported version %d", header.Version)
	}

	cfg := BuilderConfig{
		Dimension:   int(header.Dimension),
		Metric:      Metric(header.Metric),
		NumTrees:    int(header.NumTrees),
		MaxLeafSize: int(header.MaxLeaf),
	}

	count := int(header.VectorCount)
	ids := make([]int32, count)
	if count > 0 {
		if err := binary.Read(r, binary.LittleEndian, ids); err != nil {
			return nil, fmt.Errorf("annoy: read ids: %w", err)
		}
	}

	var (
		vecBacking []byte
		vecData    []float32
	)
	totalFloats := count * cfg.Dimension
	if totalFloats > 0 {
		byteLen := totalFloats * 4
		vecBacking = make([]byte, byteLen)
		if _, err := io.ReadFull(r, vecBacking); err != nil {
			return nil, fmt.Errorf("annoy: read vectors: %w", err)
		}
		vecData = unsafe.Slice((*float32)(unsafe.Pointer(&vecBacking[0])), totalFloats)
	}

	trees := make([]*node, cfg.NumTrees)
	for i := 0; i < cfg.NumTrees; i++ {
		tree, err := readTree(r, cfg.Dimension)
		if err != nil {
			return nil, fmt.Errorf("annoy: read tree %d: %w", i, err)
		}
		trees[i] = tree
	}

	return &loadedIndex{
		Config:        cfg,
		IDs:           ids,
		VectorData:    vecData,
		VectorBacking: vecBacking,
		Trees:         trees,
	}, nil
}

func readTree(r io.Reader, dim int) (*node, error) {
	flag := make([]byte, 1)
	if _, err := io.ReadFull(r, flag); err != nil {
		return nil, err
	}
	switch flag[0] {
	case nodeLeaf:
		var count uint32
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		idx := make([]int, count)
		for i := 0; i < int(count); i++ {
			var val uint32
			if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
				return nil, err
			}
			idx[i] = int(val)
		}
		return &node{
			leaf:    true,
			indices: idx,
		}, nil
	case nodeInternal:
		hp := make([]float32, dim)
		if err := binary.Read(r, binary.LittleEndian, hp); err != nil {
			return nil, err
		}
		var threshold float32
		if err := binary.Read(r, binary.LittleEndian, &threshold); err != nil {
			return nil, err
		}
		left, err := readTree(r, dim)
		if err != nil {
			return nil, err
		}
		right, err := readTree(r, dim)
		if err != nil {
			return nil, err
		}
		return &node{
			leaf:       false,
			hyperplane: hp,
			threshold:  threshold,
			left:       left,
			right:      right,
		}, nil
	default:
		return nil, fmt.Errorf("annoy: invalid tree node flag %d", flag[0])
	}
}
