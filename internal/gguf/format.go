// Package gguf provides GGUF file format parsing and memory-mapped access to tensors.
package gguf

import (
	"encoding/binary"
	"fmt"
)

// GGUF format constants
const (
	GGUFMagic   = 0x46554747 // "GGUF" in little-endian
	GGUFVersion = 3          // Current version
)

// DType represents tensor data types in GGUF
type DType uint32

const (
	DTypeF32     DType = 0
	DTypeF16     DType = 1
	DTypeQ4_0    DType = 2
	DTypeQ4_1    DType = 3
	DTypeQ5_0    DType = 6
	DTypeQ5_1    DType = 7
	DTypeQ8_0    DType = 8
	DTypeQ8_1    DType = 9
	DTypeQ2_K    DType = 10
	DTypeQ3_K    DType = 11
	DTypeQ4_K    DType = 12
	DTypeQ5_K    DType = 13
	DTypeQ6_K    DType = 14
	DTypeQ8_K    DType = 15
	DTypeI8      DType = 16
	DTypeI16     DType = 17
	DTypeI32     DType = 18
	DTypeI64     DType = 19
	DTypeF64     DType = 20
	DTypeIQ2_XXS DType = 21
	DTypeIQ2_XS  DType = 22
	DTypeIQ3_XXS DType = 23
	DTypeIQ1_S   DType = 24
	DTypeIQ4_NL  DType = 25
	DTypeIQ3_S   DType = 26
	DTypeIQ2_S   DType = 27
	DTypeIQ4_XS  DType = 28
)

// String returns the name of the data type
func (d DType) String() string {
	names := map[DType]string{
		DTypeF32:     "F32",
		DTypeF16:     "F16",
		DTypeQ4_0:    "Q4_0",
		DTypeQ4_1:    "Q4_1",
		DTypeQ5_0:    "Q5_0",
		DTypeQ5_1:    "Q5_1",
		DTypeQ8_0:    "Q8_0",
		DTypeQ8_1:    "Q8_1",
		DTypeQ2_K:    "Q2_K",
		DTypeQ3_K:    "Q3_K",
		DTypeQ4_K:    "Q4_K",
		DTypeQ5_K:    "Q5_K",
		DTypeQ6_K:    "Q6_K",
		DTypeQ8_K:    "Q8_K",
		DTypeI8:      "I8",
		DTypeI16:     "I16",
		DTypeI32:     "I32",
		DTypeI64:     "I64",
		DTypeF64:     "F64",
		DTypeIQ2_XXS: "IQ2_XXS",
		DTypeIQ2_XS:  "IQ2_XS",
		DTypeIQ3_XXS: "IQ3_XXS",
		DTypeIQ1_S:   "IQ1_S",
		DTypeIQ4_NL:  "IQ4_NL",
		DTypeIQ3_S:   "IQ3_S",
		DTypeIQ2_S:   "IQ2_S",
		DTypeIQ4_XS:  "IQ4_XS",
	}
	if name, ok := names[d]; ok {
		return name
	}
	return fmt.Sprintf("Unknown(%d)", d)
}

// BlockSize returns the block size in bytes for quantized types
func (d DType) BlockSize() int {
	switch d {
	case DTypeF32:
		return 4
	case DTypeF16:
		return 2
	case DTypeQ4_0:
		return 18 // 16 x 4-bit + 2 bytes scale
	case DTypeQ4_1:
		return 20 // 16 x 4-bit + 2 bytes scale + 2 bytes min
	case DTypeQ5_0:
		return 22 // 16 x 5-bit + 2 bytes scale
	case DTypeQ5_1:
		return 24 // 16 x 5-bit + 2 bytes scale + 2 bytes min
	case DTypeQ8_0:
		return 34 // 32 x int8 + 2 bytes scale (f16)
	case DTypeQ8_1:
		return 36 // 32 x int8 + 2 bytes scale + 2 bytes min
	case DTypeI8:
		return 1
	case DTypeI16:
		return 2
	case DTypeI32:
		return 4
	case DTypeI64:
		return 8
	case DTypeF64:
		return 8
	default:
		return 0 // Unknown or K-quants (variable)
	}
}

// ElementsPerBlock returns number of elements per quantization block
func (d DType) ElementsPerBlock() int {
	switch d {
	case DTypeF32, DTypeF16, DTypeI8, DTypeI16, DTypeI32, DTypeI64, DTypeF64:
		return 1
	case DTypeQ4_0, DTypeQ4_1, DTypeQ5_0, DTypeQ5_1:
		return 16
	case DTypeQ8_0, DTypeQ8_1:
		return 32
	default:
		return 0 // Unknown or K-quants
	}
}

// MetadataValueType represents the type of a metadata value
type MetadataValueType uint32

const (
	MetadataUint8   MetadataValueType = 0
	MetadataInt8    MetadataValueType = 1
	MetadataUint16  MetadataValueType = 2
	MetadataInt16   MetadataValueType = 3
	MetadataUint32  MetadataValueType = 4
	MetadataInt32   MetadataValueType = 5
	MetadataFloat32 MetadataValueType = 6
	MetadataBool    MetadataValueType = 7
	MetadataString  MetadataValueType = 8
	MetadataArray   MetadataValueType = 9
	MetadataUint64  MetadataValueType = 10
	MetadataInt64   MetadataValueType = 11
	MetadataFloat64 MetadataValueType = 12
)

// Header is the GGUF file header
type Header struct {
	Magic          uint32
	Version        uint32
	TensorCount    uint64
	MetadataKVSize uint64
}

// TensorInfo describes a tensor in the GGUF file
type TensorInfo struct {
	Name   string
	NDim   uint32
	Dims   []uint64
	DType  DType
	Offset uint64
}

// Metadata represents a key-value pair from GGUF metadata
type Metadata struct {
	Key   string
	Type  MetadataValueType
	Value interface{}
}

var byteOrder = binary.LittleEndian
