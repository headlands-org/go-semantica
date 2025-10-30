package annoy

import "math"

// Metric defines how distances between vectors are computed.
type Metric int

const (
	// Cosine treats vectors as L2-normalised and uses cosine distance.
	Cosine Metric = iota
	// Euclidean computes standard L2 distance.
	Euclidean
)

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	default:
		return "unknown"
	}
}

func (m Metric) distance(a, b []float32) float32 {
	switch m {
	case Cosine:
		// Expect vectors to be normalised.
		var dot float32
		for i := range a {
			dot += a[i] * b[i]
		}
		// Clamp to avoid precision issues.
		if dot > 1 {
			dot = 1
		} else if dot < -1 {
			dot = -1
		}
		return 1 - dot
	case Euclidean:
		var sum float64
		for i := range a {
			diff := float64(a[i] - b[i])
			sum += diff * diff
		}
		return float32(math.Sqrt(sum))
	default:
		panic("unsupported metric")
	}
}

func (m Metric) requiresNormalisation() bool {
	return m == Cosine
}

// Distance computes the distance based on the metric.
func (m Metric) Distance(a, b []float32) float32 {
	return m.distance(a, b)
}
