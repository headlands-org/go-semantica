package runtime

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

// float32ToFloat16 converts float32 to float16 (IEEE 754 half-precision)
// This is a test helper function - not optimized for production use
func float32ToFloat16(f32 float32) uint16 {
	bits := math.Float32bits(f32)

	sign := (bits >> 31) & 0x1
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	var f16 uint16

	if exp == 0 {
		// Zero or subnormal
		f16 = uint16(sign << 15)
	} else if exp == 0xFF {
		// Inf or NaN
		f16 = uint16(sign<<15) | 0x7C00
		if mant != 0 {
			f16 |= 0x0200 // NaN
		}
	} else {
		// Normalized
		exp_f16 := int32(exp) - 127 + 15

		if exp_f16 <= 0 {
			// Underflow to zero
			f16 = uint16(sign << 15)
		} else if exp_f16 >= 31 {
			// Overflow to infinity
			f16 = uint16(sign<<15) | 0x7C00
		} else {
			// Normal case
			mant_f16 := mant >> 13
			f16 = uint16(sign<<15) | uint16(exp_f16<<10) | uint16(mant_f16)
		}
	}

	return f16
}

// createQ8_0Block creates a Q8_0 block with the given scale and values
func createQ8_0Block(scale float32, values [32]int8) []byte {
	block := make([]byte, 34)

	// Encode scale as float16
	scaleF16 := float32ToFloat16(scale)
	binary.LittleEndian.PutUint16(block[0:2], scaleF16)

	// Copy int8 values
	for i := 0; i < 32; i++ {
		block[2+i] = byte(values[i])
	}

	return block
}

func TestExtractQ8_0Scales(t *testing.T) {
	t.Run("Basic100Blocks", func(t *testing.T) {
		// Create test data with 100 blocks
		const numBlocks = 100
		data := make([]byte, 0, numBlocks*34)
		expectedScales := make([]float32, numBlocks)

		// Create blocks with varying scales
		for i := 0; i < numBlocks; i++ {
			// Use different scales: small, medium, large, negative
			var scale float32
			switch i % 4 {
			case 0:
				scale = 0.01
			case 1:
				scale = 1.0
			case 2:
				scale = 10.0
			case 3:
				scale = -0.5
			}

			expectedScales[i] = scale

			// Create dummy int8 values (content doesn't matter for scale extraction)
			var values [32]int8
			for j := 0; j < 32; j++ {
				values[j] = int8(j - 16)
			}

			blockData := createQ8_0Block(scale, values)
			data = append(data, blockData...)
		}

		// Extract scales
		extractedScales := extractQ8_0Scales(data, numBlocks*32)

		// Verify length
		if len(extractedScales) != numBlocks {
			t.Fatalf("Expected %d scales, got %d", numBlocks, len(extractedScales))
		}

		// Verify each scale matches (with tolerance for float16 precision)
		const epsilon = 1e-3 // float16 has less precision than float32
		for i := 0; i < numBlocks; i++ {
			// Also verify against ParseQ8_0Block for consistency
			offset := i * 34
			block := gguf.ParseQ8_0Block(data[offset : offset+34])

			if math.Abs(float64(extractedScales[i]-block.Scale)) > epsilon {
				t.Errorf("Block %d: extracted scale %v doesn't match ParseQ8_0Block scale %v",
					i, extractedScales[i], block.Scale)
			}

			// Verify against expected (accounting for float16 quantization error)
			if math.Abs(float64(extractedScales[i]-expectedScales[i])) > epsilon {
				t.Errorf("Block %d: extracted scale %v doesn't match expected %v (diff: %v)",
					i, extractedScales[i], expectedScales[i],
					math.Abs(float64(extractedScales[i]-expectedScales[i])))
			}
		}
	})

	t.Run("EmptyData", func(t *testing.T) {
		data := []byte{}
		scales := extractQ8_0Scales(data, 32)

		if scales != nil {
			t.Errorf("Expected nil for empty data, got %v", scales)
		}
	})

	t.Run("PartialBlock", func(t *testing.T) {
		// Less than one full block
		data := make([]byte, 20)
		scales := extractQ8_0Scales(data, 32)

		if scales != nil {
			t.Errorf("Expected nil for partial block, got %v", scales)
		}
	})

	t.Run("SingleBlock", func(t *testing.T) {
		scale := float32(0.123)
		var values [32]int8
		for i := 0; i < 32; i++ {
			values[i] = int8(i)
		}

		data := createQ8_0Block(scale, values)
		extractedScales := extractQ8_0Scales(data, 32)

		if len(extractedScales) != 1 {
			t.Fatalf("Expected 1 scale, got %d", len(extractedScales))
		}

		const epsilon = 1e-3
		if math.Abs(float64(extractedScales[0]-scale)) > epsilon {
			t.Errorf("Scale mismatch: got %v, expected %v", extractedScales[0], scale)
		}
	})

	t.Run("ExtractMatchesParseQ8_0Block", func(t *testing.T) {
		// Create 50 blocks with random-ish scales
		const numBlocks = 50
		data := make([]byte, 0, numBlocks*34)

		scales := []float32{
			0.001, 0.01, 0.1, 1.0, 10.0,
			-0.001, -0.01, -0.1, -1.0, -10.0,
			0.5, 1.5, 2.5, 3.5, 4.5,
			-0.5, -1.5, -2.5, -3.5, -4.5,
			0.0625, 0.125, 0.25, 0.375, 0.5,
			100.0, 200.0, 300.0, 400.0, 500.0,
			-100.0, -200.0, -300.0, -400.0, -500.0,
			0.00001, 0.0001, 0.001, 0.01, 0.1,
			1000.0, 2000.0, 3000.0, 4000.0, 5000.0,
			-1000.0, -2000.0, -3000.0, -4000.0, -5000.0,
		}

		for i := 0; i < numBlocks; i++ {
			var values [32]int8
			for j := 0; j < 32; j++ {
				values[j] = int8((i * j) % 256)
			}

			blockData := createQ8_0Block(scales[i], values)
			data = append(data, blockData...)
		}

		// Extract scales
		extractedScales := extractQ8_0Scales(data, 32)

		// Verify each scale matches ParseQ8_0Block exactly
		for i := 0; i < numBlocks; i++ {
			offset := i * 34
			block := gguf.ParseQ8_0Block(data[offset : offset+34])

			if extractedScales[i] != block.Scale {
				t.Errorf("Block %d: extracted scale %v != ParseQ8_0Block scale %v",
					i, extractedScales[i], block.Scale)
			}
		}
	})

	t.Run("SpecialValues", func(t *testing.T) {
		testCases := []struct {
			name  string
			scale float32
		}{
			{"Zero", 0.0},
			{"NegativeZero", float32(math.Copysign(0, -1))},
			{"Small", 1e-4},
			{"Large", 1e4},
			{"One", 1.0},
			{"NegativeOne", -1.0},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				var values [32]int8
				data := createQ8_0Block(tc.scale, values)
				extractedScales := extractQ8_0Scales(data, 32)

				if len(extractedScales) != 1 {
					t.Fatalf("Expected 1 scale, got %d", len(extractedScales))
				}

				// Verify against ParseQ8_0Block
				block := gguf.ParseQ8_0Block(data)
				if extractedScales[0] != block.Scale {
					t.Errorf("Scale mismatch: extracted=%v, ParseQ8_0Block=%v",
						extractedScales[0], block.Scale)
				}
			})
		}
	})
}
