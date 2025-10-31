package gguf

import (
	"fmt"
	"os"
	"unsafe"

	mmap "github.com/edsrzf/mmap-go"
)

// Reader provides read access to a GGUF file via memory mapping
type Reader struct {
	path     string
	file     *os.File
	mmap     mmap.MMap // memory-mapped region (nil when using in-memory data)
	data     []byte    // backing bytes for metadata/tensor access
	header   Header
	metadata map[string]Metadata
	tensors  map[string]*TensorDesc
	dataOff  int64 // offset where tensor data begins
}

// TensorDesc describes a tensor with its location in the mapped file
type TensorDesc struct {
	Name   string
	DType  DType
	Shape  []int
	Offset int64 // file offset
	Size   int64 // size in bytes
}

// Open opens a GGUF file and memory-maps it
func Open(path string) (*Reader, error) {
	// Open file for reading
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}

	// Memory-map the file (this gives us a []byte view with zero-copy)
	mmapData, err := mmap.Map(file, mmap.RDONLY, 0)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("mmap file: %w", err)
	}

	r := &Reader{
		path:     path,
		file:     file,
		mmap:     mmapData,
		data:     mmapData,
		metadata: make(map[string]Metadata),
		tensors:  make(map[string]*TensorDesc),
	}

	// Parse header and metadata
	if err := r.parse(); err != nil {
		r.Close()
		return nil, err
	}

	return r, nil
}

// OpenBytes creates a Reader backed directly by an in-memory GGUF image.
// The provided slice must remain alive for the lifetime of the Reader.
func OpenBytes(data []byte) (*Reader, error) {
	r := &Reader{
		data:     data,
		metadata: make(map[string]Metadata),
		tensors:  make(map[string]*TensorDesc),
	}

	if err := r.parse(); err != nil {
		return nil, err
	}

	return r, nil
}

// Close closes the reader and unmaps the file
func (r *Reader) Close() error {
	var err error
	if r.mmap != nil {
		if e := r.mmap.Unmap(); e != nil {
			err = e
		}
	}
	if r.file != nil {
		if e := r.file.Close(); e != nil && err == nil {
			err = e
		}
	}
	return err
}

// parse reads the GGUF header, metadata, and tensor info
func (r *Reader) parse() error {
	// Parse metadata directly from the backing byte slice (mmap or in-memory)
	data := r.data
	offset := 0

	// Read header
	if len(data) < 24 {
		return fmt.Errorf("file too small for header")
	}

	r.header.Magic = byteOrder.Uint32(data[offset:])
	offset += 4
	if r.header.Magic != GGUFMagic {
		return fmt.Errorf("invalid magic: 0x%08x", r.header.Magic)
	}

	r.header.Version = byteOrder.Uint32(data[offset:])
	offset += 4
	if r.header.Version != GGUFVersion {
		return fmt.Errorf("unsupported version: %d", r.header.Version)
	}

	r.header.TensorCount = byteOrder.Uint64(data[offset:])
	offset += 8

	r.header.MetadataKVSize = byteOrder.Uint64(data[offset:])
	offset += 8

	// Read metadata key-value pairs
	for i := uint64(0); i < r.header.MetadataKVSize; i++ {
		md, n, err := r.readMetadata(data, offset)
		if err != nil {
			return fmt.Errorf("read metadata %d: %w", i, err)
		}
		r.metadata[md.Key] = md
		offset += n
	}

	// Read tensor info
	for i := uint64(0); i < r.header.TensorCount; i++ {
		ti, n, err := r.readTensorInfo(data, offset)
		if err != nil {
			return fmt.Errorf("read tensor info %d: %w", i, err)
		}
		offset += n

		// Convert to TensorDesc
		shape := make([]int, len(ti.Dims))
		totalElems := int64(1)
		for j, dim := range ti.Dims {
			shape[j] = int(dim)
			totalElems *= int64(dim)
		}

		// Calculate size based on dtype
		var size int64
		blockSize := ti.DType.BlockSize()
		elemsPerBlock := ti.DType.ElementsPerBlock()

		if elemsPerBlock > 0 && blockSize > 0 {
			numBlocks := (totalElems + int64(elemsPerBlock) - 1) / int64(elemsPerBlock)
			size = numBlocks * int64(blockSize)
		} else {
			size = totalElems * int64(blockSize)
		}

		desc := &TensorDesc{
			Name:   ti.Name,
			DType:  ti.DType,
			Shape:  shape,
			Offset: int64(ti.Offset),
			Size:   size,
		}
		r.tensors[ti.Name] = desc
	}

	// Align to 32-byte boundary for tensor data
	r.dataOff = int64(align(offset, 32))

	return nil
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// readMetadata reads a single metadata key-value pair
func (r *Reader) readMetadata(data []byte, offset int) (Metadata, int, error) {
	start := offset
	md := Metadata{}

	// Read key (string)
	keyLen := byteOrder.Uint64(data[offset:])
	offset += 8
	md.Key = string(data[offset : offset+int(keyLen)])
	offset += int(keyLen)

	// Read value type
	md.Type = MetadataValueType(byteOrder.Uint32(data[offset:]))
	offset += 4

	// Read value based on type
	var err error
	md.Value, offset, err = r.readMetadataValue(data, offset, md.Type)
	if err != nil {
		return md, 0, err
	}

	return md, offset - start, nil
}

// readMetadataValue reads a metadata value
func (r *Reader) readMetadataValue(data []byte, offset int, typ MetadataValueType) (interface{}, int, error) {
	switch typ {
	case MetadataUint8:
		return data[offset], offset + 1, nil
	case MetadataInt8:
		return int8(data[offset]), offset + 1, nil
	case MetadataUint16:
		return byteOrder.Uint16(data[offset:]), offset + 2, nil
	case MetadataInt16:
		return int16(byteOrder.Uint16(data[offset:])), offset + 2, nil
	case MetadataUint32:
		return byteOrder.Uint32(data[offset:]), offset + 4, nil
	case MetadataInt32:
		return int32(byteOrder.Uint32(data[offset:])), offset + 4, nil
	case MetadataFloat32:
		bits := byteOrder.Uint32(data[offset:])
		return *(*float32)(unsafe.Pointer(&bits)), offset + 4, nil
	case MetadataUint64:
		return byteOrder.Uint64(data[offset:]), offset + 8, nil
	case MetadataInt64:
		return int64(byteOrder.Uint64(data[offset:])), offset + 8, nil
	case MetadataFloat64:
		bits := byteOrder.Uint64(data[offset:])
		return *(*float64)(unsafe.Pointer(&bits)), offset + 8, nil
	case MetadataBool:
		return data[offset] != 0, offset + 1, nil
	case MetadataString:
		strlen := byteOrder.Uint64(data[offset:])
		offset += 8
		str := string(data[offset : offset+int(strlen)])
		return str, offset + int(strlen), nil
	case MetadataArray:
		// Read array type
		arrType := MetadataValueType(byteOrder.Uint32(data[offset:]))
		offset += 4
		// Read array length
		arrLen := byteOrder.Uint64(data[offset:])
		offset += 8
		// Read array elements
		arr := make([]interface{}, arrLen)
		for i := uint64(0); i < arrLen; i++ {
			var err error
			arr[i], offset, err = r.readMetadataValue(data, offset, arrType)
			if err != nil {
				return nil, offset, err
			}
		}
		return arr, offset, nil
	default:
		return nil, offset, fmt.Errorf("unknown metadata type: %d", typ)
	}
}

// readTensorInfo reads tensor information
func (r *Reader) readTensorInfo(data []byte, offset int) (TensorInfo, int, error) {
	start := offset
	ti := TensorInfo{}

	// Read name
	nameLen := byteOrder.Uint64(data[offset:])
	offset += 8
	ti.Name = string(data[offset : offset+int(nameLen)])
	offset += int(nameLen)

	// Read number of dimensions
	ti.NDim = byteOrder.Uint32(data[offset:])
	offset += 4

	// Read dimensions
	ti.Dims = make([]uint64, ti.NDim)
	for i := uint32(0); i < ti.NDim; i++ {
		ti.Dims[i] = byteOrder.Uint64(data[offset:])
		offset += 8
	}

	// Read dtype
	ti.DType = DType(byteOrder.Uint32(data[offset:]))
	offset += 4

	// Read offset
	ti.Offset = byteOrder.Uint64(data[offset:])
	offset += 8

	return ti, offset - start, nil
}

// GetMetadata returns metadata value by key
func (r *Reader) GetMetadata(key string) (interface{}, bool) {
	md, ok := r.metadata[key]
	if !ok {
		return nil, false
	}
	return md.Value, true
}

// GetTensor returns tensor descriptor by name
func (r *Reader) GetTensor(name string) (*TensorDesc, bool) {
	desc, ok := r.tensors[name]
	return desc, ok
}

// ListTensors returns all tensor names
func (r *Reader) ListTensors() []string {
	names := make([]string, 0, len(r.tensors))
	for name := range r.tensors {
		names = append(names, name)
	}
	return names
}

// GetTensorData returns a view of the tensor data as a byte slice
// The returned slice is a direct view into the backing data (zero-copy)
func (r *Reader) GetTensorData(name string) ([]byte, error) {
	desc, ok := r.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}

	offset := r.dataOff + desc.Offset
	end := offset + desc.Size
	if offset < 0 || end > int64(len(r.data)) {
		return nil, fmt.Errorf("tensor data out of bounds: %s (offset=%d size=%d fileSize=%d)",
			name, offset, desc.Size, len(r.data))
	}

	return r.data[offset:end], nil
}

// Header returns the GGUF header
func (r *Reader) Header() Header {
	return r.header
}

// align rounds up to the nearest multiple of alignment
func align(offset, alignment int) int {
	return (offset + alignment - 1) &^ (alignment - 1)
}
