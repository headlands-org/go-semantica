//go:build amd64

package brute

func dotInt8AVX2(vec *int8, query *float32, length int) float32
func dotInt16AVX2(vec *int16, query *float32, length int) float32
func dotFloat32AVX2(vec *float32, query *float32, length int) float32
