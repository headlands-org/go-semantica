// Package annoy implements a pure-Go approximation of Spotify's Annoy index.
// It provides a high-level builder and search API tightly integrated with the
// EmbeddingGemma runtime shipped in this repository. The index can be built,
// serialized to a compact binary blob, mmap'd back, and queried for semantic
// similarity with cosine or Euclidean distance.
package annoy
