package annoy

import (
	"math"
	"math/rand"
)

type node struct {
	leaf       bool
	indices    []int
	hyperplane []float32
	threshold  float32
	left       *node
	right      *node
}

func buildNode(indices []int, vectors [][]float32, cfg BuilderConfig, rng *rand.Rand) *node {
	if len(indices) <= cfg.MaxLeafSize {
		leafIdx := make([]int, len(indices))
		copy(leafIdx, indices)
		return &node{
			leaf:    true,
			indices: leafIdx,
		}
	}

	// Sample two random points to define a hyperplane.
	var (
		aIdx = indices[rng.Intn(len(indices))]
		bIdx = indices[rng.Intn(len(indices))]
	)
	if aIdx == bIdx && len(indices) > 1 {
		bIdx = indices[(rng.Intn(len(indices)-1)+1)%len(indices)]
	}

	vecA := vectors[aIdx]
	vecB := vectors[bIdx]
	dim := len(vecA)
	normal := make([]float32, dim)
	for i := 0; i < dim; i++ {
		normal[i] = vecB[i] - vecA[i]
	}

	// If vectors are identical, fall back to random normal.
	if magnitude(normal) == 0 {
		for i := range normal {
			normal[i] = rng.Float32()*2 - 1
		}
	}
	normalise(normal)

	mid := make([]float32, dim)
	for i := 0; i < dim; i++ {
		mid[i] = (vecA[i] + vecB[i]) * 0.5
	}
	threshold := dot(normal, mid)

	leftIdx := make([]int, 0, len(indices)/2)
	rightIdx := make([]int, 0, len(indices)/2)
	for _, idx := range indices {
		score := dot(normal, vectors[idx])
		if score <= threshold {
			leftIdx = append(leftIdx, idx)
		} else {
			rightIdx = append(rightIdx, idx)
		}
	}

	// Guard against degenerate splits.
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		leafIdx := make([]int, len(indices))
		copy(leafIdx, indices)
		return &node{
			leaf:    true,
			indices: leafIdx,
		}
	}

	return &node{
		leaf:       false,
		hyperplane: normal,
		threshold:  threshold,
		left:       buildNode(leftIdx, vectors, cfg, rng),
		right:      buildNode(rightIdx, vectors, cfg, rng),
	}
}

func magnitude(vec []float32) float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	return float32(math.Sqrt(sum))
}

func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
