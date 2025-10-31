package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// TensorView provides typed access to tensor data
type TensorView struct {
	desc *TensorDesc
	data []byte
}

// NewTensorView creates a view over tensor data
func NewTensorView(desc *TensorDesc, data []byte) *TensorView {
	return &TensorView{
		desc: desc,
		data: data,
	}
}

// Shape returns the tensor shape
func (tv *TensorView) Shape() []int {
	return tv.desc.Shape
}

// DType returns the tensor data type
func (tv *TensorView) DType() DType {
	return tv.desc.DType
}

// NumElements returns total number of elements
func (tv *TensorView) NumElements() int {
	n := 1
	for _, d := range tv.desc.Shape {
		n *= d
	}
	return n
}

// AsFloat32 returns tensor data as []float32 (for F32 tensors)
func (tv *TensorView) AsFloat32() ([]float32, error) {
	if tv.desc.DType != DTypeF32 {
		return nil, fmt.Errorf("tensor is not F32: %s", tv.desc.DType)
	}

	n := tv.NumElements()
	if len(tv.data) < n*4 {
		return nil, fmt.Errorf("insufficient data for F32 tensor")
	}

	// Create slice view using unsafe
	return unsafe.Slice((*float32)(unsafe.Pointer(&tv.data[0])), n), nil
}

// AsInt8 returns tensor data as []int8 (for I8 tensors)
func (tv *TensorView) AsInt8() ([]int8, error) {
	if tv.desc.DType != DTypeI8 {
		return nil, fmt.Errorf("tensor is not I8: %s", tv.desc.DType)
	}

	n := tv.NumElements()
	if len(tv.data) < n {
		return nil, fmt.Errorf("insufficient data for I8 tensor")
	}

	return unsafe.Slice((*int8)(unsafe.Pointer(&tv.data[0])), n), nil
}

// AsInt32 returns tensor data as []int32 (for I32 tensors)
func (tv *TensorView) AsInt32() ([]int32, error) {
	if tv.desc.DType != DTypeI32 {
		return nil, fmt.Errorf("tensor is not I32: %s", tv.desc.DType)
	}

	n := tv.NumElements()
	if len(tv.data) < n*4 {
		return nil, fmt.Errorf("insufficient data for I32 tensor")
	}

	return unsafe.Slice((*int32)(unsafe.Pointer(&tv.data[0])), n), nil
}

// Q8_0Block represents a Q8_0 quantization block
// 32 int8 values + 1 float16 scale
type Q8_0Block struct {
	Scale float32  // dequantized from f16
	Qs    [32]int8 // quantized values
}

// ParseQ8_0Block parses a Q8_0 block from bytes
func ParseQ8_0Block(data []byte) Q8_0Block {
	if len(data) < 34 {
		return Q8_0Block{}
	}

	block := Q8_0Block{}

	// Read scale as float16, convert to float32
	scaleF16 := binary.LittleEndian.Uint16(data[0:2])
	block.Scale = float16ToFloat32(scaleF16)

	// Read 32 int8 values
	for i := 0; i < 32; i++ {
		block.Qs[i] = int8(data[2+i])
	}

	return block
}

// Dequantize dequantizes the block to float32 values
func (b *Q8_0Block) Dequantize(dst []float32) {
	for i := 0; i < 32; i++ {
		dst[i] = float32(b.Qs[i]) * b.Scale
	}
}

// Q8_0Iterator iterates over Q8_0 blocks
type Q8_0Iterator struct {
	data     []byte
	offset   int
	numElems int
	elemIdx  int
}

// NewQ8_0Iterator creates an iterator for Q8_0 quantized data
func NewQ8_0Iterator(data []byte, numElems int) *Q8_0Iterator {
	return &Q8_0Iterator{
		data:     data,
		numElems: numElems,
	}
}

// Next returns the next block, or false if done
func (it *Q8_0Iterator) Next() (Q8_0Block, bool) {
	if it.elemIdx >= it.numElems {
		return Q8_0Block{}, false
	}

	if it.offset+34 > len(it.data) {
		return Q8_0Block{}, false
	}

	block := ParseQ8_0Block(it.data[it.offset:])
	it.offset += 34
	it.elemIdx += 32

	return block, true
}

// DequantizeQ8_0Row dequantizes a single row of Q8_0 data into dst
// Used for on-the-fly token embedding lookup (zero-copy architecture)
// rowData: Q8_0 bytes for this row (numCols/32 blocks * 34 bytes)
// dst: pre-allocated float32 slice to write to (length numCols)
func DequantizeQ8_0Row(dst []float32, rowData []byte, numCols int) {
	blocksInRow := (numCols + 31) / 32
	bytesPerBlock := 34

	for blockIdx := 0; blockIdx < blocksInRow; blockIdx++ {
		offset := blockIdx * bytesPerBlock
		block := ParseQ8_0Block(rowData[offset : offset+bytesPerBlock])

		// Determine how many elements in this block
		blockStart := blockIdx * 32
		blockEnd := blockStart + 32
		if blockEnd > numCols {
			blockEnd = numCols
		}

		// Dequantize directly into dst
		for i := blockStart; i < blockEnd; i++ {
			dst[i] = float32(block.Qs[i-blockStart]) * block.Scale
		}
	}
}

// DequantizeQ8_0 dequantizes an entire Q8_0 tensor to float32
func DequantizeQ8_0(data []byte, numElems int) []float32 {
	result := make([]float32, numElems)
	it := NewQ8_0Iterator(data, numElems)

	idx := 0
	var tmp [32]float32
	for {
		block, ok := it.Next()
		if !ok {
			break
		}

		block.Dequantize(tmp[:])

		// Copy to result (handle partial blocks at the end)
		remaining := numElems - idx
		if remaining > 32 {
			remaining = 32
		}
		copy(result[idx:idx+remaining], tmp[:remaining])
		idx += remaining
	}

	return result
}

// float16ToFloat32 converts float16 to float32
func float16ToFloat32(f16 uint16) float32 {
	// Extract sign, exponent, mantissa
	sign := (f16 >> 15) & 0x1
	exp := (f16 >> 10) & 0x1F
	mant := f16 & 0x3FF

	var f32Bits uint32

	if exp == 0 {
		if mant == 0 {
			// Zero
			f32Bits = uint32(sign) << 31
		} else {
			// Subnormal - convert to normalized float32
			exp := uint32(127 - 15)
			mant := uint32(mant)
			// Normalize
			for (mant & 0x400) == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f32Bits = (uint32(sign) << 31) | (exp << 23) | (mant << 13)
		}
	} else if exp == 0x1F {
		// Inf or NaN
		f32Bits = (uint32(sign) << 31) | (0xFF << 23) | (uint32(mant) << 13)
	} else {
		// Normalized
		f32Bits = (uint32(sign) << 31) | ((uint32(exp-15+127) & 0xFF) << 23) | (uint32(mant) << 13)
	}

	return math.Float32frombits(f32Bits)
}
