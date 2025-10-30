package brute

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	"github.com/lth/pure-go-llamas/search"
)

var bruteMagic = [4]byte{'B', 'R', 'U', 'T'}

const bruteVersion uint16 = 1

type bruteHeader struct {
	Magic        [4]byte
	Version      uint16
	Dimension    uint16
	Count        uint32
	Quantization uint8
	_            [7]byte
}

// Serializer implements search.Serializer for brute-force indices.
type Serializer struct{}

// Serialize encodes the index.
func (Serializer) Serialize(idx search.Index) ([]byte, error) {
	bruteIdx, ok := idx.(*Index)
	if !ok {
		return nil, fmt.Errorf("brute: serializer expects *Index, got %T", idx)
	}
	var buf bytes.Buffer
	if err := writeIndex(&buf, bruteIdx); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Deserialize decodes an index from bytes.
func (Serializer) Deserialize(data []byte) (search.Index, error) {
	return readIndex(bytes.NewReader(data))
}

func writeIndex(w io.Writer, idx *Index) error {
	hdr := bruteHeader{
		Magic:        bruteMagic,
		Version:      bruteVersion,
		Dimension:    uint16(idx.dimension),
		Count:        uint32(len(idx.ids)),
		Quantization: uint8(idx.quantization),
	}
	if err := binary.Write(w, binary.LittleEndian, hdr); err != nil {
		return err
	}

	if len(idx.ids) > 0 {
		if err := binary.Write(w, binary.LittleEndian, idx.ids); err != nil {
			return err
		}
	}

	switch idx.quantization {
	case QuantizeFloat32:
		if err := binary.Write(w, binary.LittleEndian, idx.float32Data); err != nil {
			return err
		}
	case QuantizeInt8:
		if err := binary.Write(w, binary.LittleEndian, idx.int8Data); err != nil {
			return err
		}
	case QuantizeInt16:
		if err := binary.Write(w, binary.LittleEndian, idx.int16Data); err != nil {
			return err
		}
	default:
		return fmt.Errorf("brute: unsupported quantization %d", idx.quantization)
	}

	return nil
}

func readIndex(r io.Reader) (*Index, error) {
	var hdr bruteHeader
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr.Magic != bruteMagic {
		return nil, fmt.Errorf("brute: invalid magic: %q", hdr.Magic)
	}
	if hdr.Version != bruteVersion {
		return nil, fmt.Errorf("brute: unsupported version %d", hdr.Version)
	}

	count := int(hdr.Count)
	idx := &Index{
		dimension:    int(hdr.Dimension),
		quantization: QuantizationMode(hdr.Quantization),
		ids:          make([]int32, count),
		idToIdx:      make(map[int32]int, count),
	}

	if count > 0 {
		if err := binary.Read(r, binary.LittleEndian, idx.ids); err != nil {
			return nil, err
		}
		for i, id := range idx.ids {
			idx.idToIdx[id] = i
		}
	}

	switch idx.quantization {
	case QuantizeFloat32:
		byteLen := count * idx.dimension * 4
		if byteLen > 0 {
			buf := make([]byte, byteLen)
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, err
			}
			idx.float32Base = buf
			idx.float32Data = unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), count*idx.dimension)
		}
	case QuantizeInt8:
		byteLen := count * idx.dimension
		buf := make([]byte, byteLen)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		idx.int8Base = buf
		if byteLen > 0 {
			slice := unsafe.Slice((*int8)(unsafe.Pointer(&buf[0])), byteLen)
			idx.int8Data = slice
		}
	case QuantizeInt16:
		byteLen := count * idx.dimension * 2
		buf := make([]byte, byteLen)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		idx.int16Base = buf
		if byteLen > 0 {
			length := byteLen / 2
			slice := unsafe.Slice((*int16)(unsafe.Pointer(&buf[0])), length)
			idx.int16Data = slice
		}
	default:
		return nil, fmt.Errorf("brute: unsupported quantization %d", hdr.Quantization)
	}

	return idx, nil
}
