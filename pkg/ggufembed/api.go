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
	// EmbedInputs generates embeddings for the given inputs, applying the
	// appropriate prompt for each task.
	EmbedInputs(ctx context.Context, inputs []EmbedInput) ([][]float32, error)

	// Embed generates embeddings for the given texts using TaskSearchQuery prompts.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedSingleInput embeds a single task-aware input.
	EmbedSingleInput(ctx context.Context, input EmbedInput) ([]float32, error)

	// EmbedSingle generates an embedding for a single text using TaskSearchQuery prompts.
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
	// NumThreads specifies the number of worker goroutines for parallel text processing.
	//
	// If set to 0 (default): Auto-tuned based on batch size using an adaptive strategy:
	//   - Batch 1-4:   1 worker (serial, avoids goroutine overhead)
	//   - Batch 5-16:  min(batch, NumCPU/4) workers (light parallelism)
	//   - Batch 17-32: min(batch, NumCPU/2) workers (moderate parallelism)
	//   - Batch 33+:   min(batch, NumCPU) workers (full CPU utilization)
	//
	// If set > 0: Uses the specified number of workers (no auto-tuning).
	//
	// Default: 0 (auto-tune)
	NumThreads int

	// BatchSize specifies the batch size for parallel processing (default: 1)
	// This controls how many texts are grouped together in a single goroutine.
	// Larger batch sizes reduce goroutine overhead but increase latency per batch.
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
	return openWithLoader(func() (*modelrt.Model, error) {
		return modelrt.LoadModel(path)
	}, opts...)
}

// OpenBytes loads a GGUF model directly from an in-memory byte slice.
// The caller must ensure the slice remains valid for the lifetime of the runtime.
func OpenBytes(data []byte, opts ...Option) (Runtime, error) {
	return openWithLoader(func() (*modelrt.Model, error) {
		return modelrt.LoadModelFromBytes(data)
	}, opts...)
}

func openWithLoader(loader func() (*modelrt.Model, error), opts ...Option) (Runtime, error) {
	options := Options{
		NumThreads: 0, // 0 = auto-tune based on batch size
		BatchSize:  1,
		Verbose:    false,
	}

	for _, opt := range opts {
		opt(&options)
	}

	model, err := loader()
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

	inputs := make([]EmbedInput, len(texts))
	for i, text := range texts {
		inputs[i] = EmbedInput{
			Task:    TaskSearchQuery,
			Content: text,
		}
	}
	return r.EmbedInputs(ctx, inputs)
}

// EmbedSingle generates an embedding for a single text using TaskSearchQuery prompts.
func (r *embedRuntime) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	return r.EmbedSingleInput(ctx, EmbedInput{
		Task:    TaskSearchQuery,
		Content: text,
	})
}

func (r *embedRuntime) EmbedInputs(ctx context.Context, inputs []EmbedInput) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	prompts := make([]string, len(inputs))
	for i, in := range inputs {
		prompt, err := in.prompt()
		if err != nil {
			return nil, fmt.Errorf("build prompt for input %d: %w", i, err)
		}
		prompts[i] = prompt
	}

	return r.embedPrompts(ctx, prompts)
}

func (r *embedRuntime) EmbedSingleInput(ctx context.Context, input EmbedInput) ([]float32, error) {
	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	prompt, err := input.prompt()
	if err != nil {
		return nil, err
	}

	return r.embedPrompt(prompt)
}

// embedSingle is the internal implementation
func (r *embedRuntime) embedPrompt(text string) ([]float32, error) {
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

func (r *embedRuntime) embedPrompts(ctx context.Context, prompts []string) ([][]float32, error) {
	if len(prompts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(prompts))
	errors := make([]error, len(prompts))

	workers := runtime.NumCPU()
	if len(prompts) < workers {
		workers = len(prompts)
	}

	promptsPerWorker := (len(prompts) + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * promptsPerWorker
		end := start + promptsPerWorker
		if end > len(prompts) {
			end = len(prompts)
		}
		if start >= len(prompts) {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			for i := start; i < end; i++ {
				select {
				case <-ctx.Done():
					errors[i] = ctx.Err()
					return
				default:
				}

				embedding, err := r.embedPrompt(prompts[i])
				if err != nil {
					errors[i] = err
				} else {
					results[i] = embedding
				}
			}
		}(start, end)
	}

	wg.Wait()

	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
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
