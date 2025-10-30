package annoy

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
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

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

func writeIndex(w io.Writer, idx *serialisableIndex) (int64, error) {
	cw := &countingWriter{w: w}
	header := fileHeader{
		Magic:       fileMagic,
		Version:     fileVersion,
		Metric:      uint16(idx.Config.Metric),
		Dimension:   uint16(idx.Config.Dimension),
		NumTrees:    uint16(len(idx.Trees)),
		MaxLeaf:     uint16(idx.Config.MaxLeafSize),
		VectorCount: uint32(len(idx.Vectors)),
	}
	if header.Dimension == 0 {
		header.Dimension = uint16(len(idx.Vectors[0]))
	}

	if header.NumTrees == 0 {
		header.NumTrees = 0
	}

	if err := binary.Write(cw, binary.LittleEndian, header); err != nil {
		return cw.n, err
	}

	dim := int(header.Dimension)
	for i := range idx.Vectors {
		idBytes := []byte(idx.IDs[i])
		if len(idBytes) > mathMaxUint16 {
			return cw.n, fmt.Errorf("annoy: id %q exceeds length limit", idx.IDs[i])
		}
		meta := idx.Metadata[i]
		if err := binary.Write(cw, binary.LittleEndian, uint16(len(idBytes))); err != nil {
			return cw.n, err
		}
		if _, err := cw.Write(idBytes); err != nil {
			return cw.n, err
		}
		if err := binary.Write(cw, binary.LittleEndian, uint32(len(meta))); err != nil {
			return cw.n, err
		}
		if len(meta) > 0 {
			if _, err := cw.Write(meta); err != nil {
				return cw.n, err
			}
		}
		if err := binary.Write(cw, binary.LittleEndian, idx.Vectors[i]); err != nil {
			return cw.n, err
		}
	}

	for _, tree := range idx.Trees {
		if err := writeTree(cw, tree, dim); err != nil {
			return cw.n, err
		}
	}
	return cw.n, nil
}

const (
	nodeLeaf      = byte(1)
	nodeInternal  = byte(0)
	mathMaxUint16 = 1<<16 - 1
)

func writeTree(w io.Writer, n *node, dim int) error {
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
	if err := writeTree(w, n.left, dim); err != nil {
		return err
	}
	return writeTree(w, n.right, dim)
}

type loadedIndex struct {
	Config   BuilderConfig
	IDs      []string
	Vectors  [][]float32
	Metadata [][]byte
	Trees    []*node
}

func readIndex(data []byte) (*loadedIndex, error) {
	r := bytes.NewReader(data)
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
	ids := make([]string, count)
	vectors := make([][]float32, count)
	metadata := make([][]byte, count)

	for i := 0; i < count; i++ {
		var idLen uint16
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return nil, fmt.Errorf("annoy: read id length: %w", err)
		}
		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(r, idBytes); err != nil {
			return nil, fmt.Errorf("annoy: read id: %w", err)
		}
		ids[i] = string(idBytes)

		var metaLen uint32
		if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
			return nil, fmt.Errorf("annoy: read metadata length: %w", err)
		}
		if metaLen > 0 {
			meta := make([]byte, metaLen)
			if _, err := io.ReadFull(r, meta); err != nil {
				return nil, fmt.Errorf("annoy: read metadata: %w", err)
			}
			metadata[i] = meta
		}
		vec := make([]float32, cfg.Dimension)
		if err := binary.Read(r, binary.LittleEndian, vec); err != nil {
			return nil, fmt.Errorf("annoy: read vector: %w", err)
		}
		vectors[i] = vec
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
		Config:   cfg,
		IDs:      ids,
		Vectors:  vectors,
		Metadata: metadata,
		Trees:    trees,
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
