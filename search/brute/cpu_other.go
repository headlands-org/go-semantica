//go:build !amd64

package brute

const (
	hasAVX2 = false
	hasFMA  = false
)
