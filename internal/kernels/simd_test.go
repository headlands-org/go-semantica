package kernels

import (
	"testing"
)

func TestDotProductINT8(t *testing.T) {
	tests := []struct {
		name string
		a    []int8
		b    []int8
		want int32
	}{
		{
			name: "simple_positive",
			a:    []int8{1, 2, 3, 4},
			b:    []int8{5, 6, 7, 8},
			want: 1*5 + 2*6 + 3*7 + 4*8, // 70
		},
		{
			name: "with_negatives",
			a:    []int8{-1, 2, -3, 4},
			b:    []int8{5, -6, 7, -8},
			want: -1*5 + 2*(-6) + (-3)*7 + 4*(-8), // -70
		},
		{
			name: "32_elements",
			a:    make([]int8, 32),
			b:    make([]int8, 32),
			want: 32, // Will set all to 1 * 1
		},
		{
			name: "64_elements",
			a:    make([]int8, 64),
			b:    make([]int8, 64),
			want: 64, // Will set all to 1 * 1
		},
		{
			name: "33_elements_not_aligned",
			a:    make([]int8, 33),
			b:    make([]int8, 33),
			want: 33,
		},
	}

	// Fill test arrays with 1s for the larger tests
	for i := range tests {
		if tests[i].name == "32_elements" || tests[i].name == "64_elements" || tests[i].name == "33_elements_not_aligned" {
			for j := range tests[i].a {
				tests[i].a[j] = 1
				tests[i].b[j] = 1
			}
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test SIMD version
			got := dotProductINT8SIMD(tt.a, tt.b, len(tt.a))
			if got != tt.want {
				t.Errorf("dotProductINT8SIMD() = %v, want %v", got, tt.want)
			}

			// Test assembly version directly
			if len(tt.a) >= 32 {
				gotAsm := dotProductINT8Asm(&tt.a[0], &tt.b[0], len(tt.a))
				if gotAsm != tt.want {
					t.Errorf("dotProductINT8Asm() = %v, want %v", gotAsm, tt.want)
				}
			}
		})
	}
}

func BenchmarkDotProductINT8(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512, 768, 1024}

	for _, size := range sizes {
		a := make([]int8, size)
		bb := make([]int8, size)
		for i := range a {
			a[i] = int8(i % 127)
			bb[i] = int8((i + 1) % 127)
		}

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				_ = dotProductINT8SIMD(a, bb, size)
			}
		})
	}
}
