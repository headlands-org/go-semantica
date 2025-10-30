package brute

import "math"

// QuantizationMode controls how vectors are stored on disk and in memory.
type QuantizationMode uint8

const (
	// QuantizeFloat32 stores vectors as raw float32 values.
	QuantizeFloat32 QuantizationMode = 0
	// QuantizeInt8 stores vectors as signed int8 values in the range [-127, 127].
	QuantizeInt8 QuantizationMode = 1
	// QuantizeInt16 stores vectors as signed int16 values in the range [-32767, 32767].
	QuantizeInt16 QuantizationMode = 2
)

func quantizeInt8(vec []float32) []int8 {
	out := make([]int8, len(vec))
	for i, v := range vec {
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		out[i] = int8(math.Round(float64(v) * 127))
	}
	return out
}

func quantizeInt16(vec []float32) []int16 {
	out := make([]int16, len(vec))
	for i, v := range vec {
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		out[i] = int16(math.Round(float64(v) * 32767))
	}
	return out
}

func dequantizeInt8(src []int8, dst []float32) {
	const scale = 1.0 / 127.0
	for i, v := range src {
		dst[i] = float32(v) * scale
	}
}

func dequantizeInt16(src []int16, dst []float32) {
	const scale = 1.0 / 32767.0
	for i, v := range src {
		dst[i] = float32(v) * scale
	}
}

var int8ToFloat32 = func() [256]float32 {
	const scale = 1.0 / 127.0
	var lut [256]float32
	for i := 0; i < 256; i++ {
		lut[i] = float32(int8(i-128)) * scale
	}
	return lut
}()

func dotInt8(vec []int8, query []float32, dim int) float32 {
	sum0, sum1, sum2, sum3 := float32(0), float32(0), float32(0), float32(0)
	i := 0
	for ; i+4 <= dim; i += 4 {
		sum0 += query[i] * int8ToFloat32[uint8(vec[i])]
		sum1 += query[i+1] * int8ToFloat32[uint8(vec[i+1])]
		sum2 += query[i+2] * int8ToFloat32[uint8(vec[i+2])]
		sum3 += query[i+3] * int8ToFloat32[uint8(vec[i+3])]
	}
	sum := sum0 + sum1 + sum2 + sum3
	for ; i < dim; i++ {
		sum += query[i] * int8ToFloat32[uint8(vec[i])]
	}
	return sum
}

func dotInt16(vec []int16, query []float32, dim int) float32 {
	const scale = 1.0 / 32767.0
	sum := float32(0)
	for i := 0; i < dim; i++ {
		sum += query[i] * float32(vec[i]) * scale
	}
	return sum
}
