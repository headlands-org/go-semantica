// Package ggufembed provides a high-level API for GGUF embedding models
package ggufembed

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	modelrt "github.com/lth/pure-go-llamas/internal/runtime"
)

// Runtime is the main interface for the embedding runtime
type Runtime interface {
	// Embed generates embeddings for the given texts
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedSingle generates an embedding for a single text
	EmbedSingle(ctx context.Context, text string) ([]float32, error)

	// Close releases resources
	Close() error

	// EmbedDim returns the embedding dimension
	EmbedDim() int

	// MaxSeqLen returns the maximum sequence length
	MaxSeqLen() int
}

// embedRuntime implements Runtime
type embedRuntime struct {
	model   *modelrt.Model
	options Options
	pool    *sync.Pool // pool of worker resources
}

// Options configures the runtime
type Options struct {
	// NumThreads specifies the number of threads to use (default: runtime.NumCPU())
	NumThreads int

	// BatchSize specifies the batch size for parallel processing (default: 1)
	BatchSize int

	// Verbose enables verbose logging
	Verbose bool
}

// Option is a functional option for configuring the runtime
type Option func(*Options)

// WithThreads sets the number of threads
func WithThreads(n int) Option {
	return func(o *Options) {
		o.NumThreads = n
	}
}

// WithBatchSize sets the batch size
func WithBatchSize(n int) Option {
	return func(o *Options) {
		o.BatchSize = n
	}
}

// WithVerbose enables verbose logging
func WithVerbose(v bool) Option {
	return func(o *Options) {
		o.Verbose = v
	}
}

// Open opens a GGUF model file and returns a Runtime
func Open(path string, opts ...Option) (Runtime, error) {
	options := Options{
		NumThreads: runtime.NumCPU(),
		BatchSize:  1,
		Verbose:    false,
	}

	for _, opt := range opts {
		opt(&options)
	}

	// Load model
	model, err := modelrt.LoadModel(path)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	rt := &embedRuntime{
		model:   model,
		options: options,
		pool: &sync.Pool{
			New: func() interface{} {
				return &workerState{}
			},
		},
	}

	return rt, nil
}

// workerState holds per-worker scratch space
type workerState struct {
	// Can add worker-specific buffers here if needed
}

// Embed generates embeddings for multiple texts
func (r *embedRuntime) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	errors := make([]error, len(texts))

	// Process in batches
	batchSize := r.options.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, r.options.NumThreads)

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			sem <- struct{}{}        // Acquire
			defer func() { <-sem }() // Release

			for j := start; j < end; j++ {
				// Check context
				select {
				case <-ctx.Done():
					errors[j] = ctx.Err()
					return
				default:
				}

				// Encode
				embedding, err := r.embedSingle(texts[j])
				if err != nil {
					errors[j] = err
				} else {
					results[j] = embedding
				}
			}
		}(i, end)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
}

// EmbedSingle generates an embedding for a single text
func (r *embedRuntime) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return r.embedSingle(text)
}

// embedSingle is the internal implementation
func (r *embedRuntime) embedSingle(text string) ([]float32, error) {
	// Tokenize
	tokenIDs, err := r.model.Tokenizer().Encode(text)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}

	// Check length
	if len(tokenIDs) > r.model.Config().MaxSeqLen {
		return nil, fmt.Errorf("sequence too long: %d > %d", len(tokenIDs), r.model.Config().MaxSeqLen)
	}

	// Forward pass
	embedding, err := r.model.Forward(tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}

	return embedding, nil
}

// Close releases resources
func (r *embedRuntime) Close() error {
	return r.model.Close()
}

// EmbedDim returns the embedding dimension
func (r *embedRuntime) EmbedDim() int {
	return r.model.Config().EmbedDim
}

// MaxSeqLen returns the maximum sequence length
func (r *embedRuntime) MaxSeqLen() int {
	return r.model.Config().MaxSeqLen
}
