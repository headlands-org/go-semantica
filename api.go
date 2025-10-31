package semantica

import (
	"context"
	"fmt"

	"github.com/headlands-org/go-semantica/pkg/ggufembed"
)

// Task describes the embedding task to run (prepending the correct prompt).
type Task = ggufembed.Task

const (
	TaskNone               = ggufembed.TaskNone
	TaskSearchQuery        = ggufembed.TaskSearchQuery
	TaskSearchDocument     = ggufembed.TaskSearchDocument
	TaskQuestionAnswering  = ggufembed.TaskQuestionAnswering
	TaskFactVerification   = ggufembed.TaskFactVerification
	TaskClassification     = ggufembed.TaskClassification
	TaskClustering         = ggufembed.TaskClustering
	TaskSemanticSimilarity = ggufembed.TaskSemanticSimilarity
	TaskCodeRetrieval      = ggufembed.TaskCodeRetrieval
)

// Dimensions represents the number of dimensions to return.
type Dimensions int

const (
	DimensionsAuto    Dimensions = 0
	Dimensions768     Dimensions = 768
	Dimensions512     Dimensions = 512
	Dimensions256     Dimensions = 256
	Dimensions128     Dimensions = 128
	DimensionsDefault Dimensions = Dimensions(ggufembed.DefaultEmbedDim)
	DefaultEmbedDim              = ggufembed.DefaultEmbedDim
)

// Option configures the runtime.
type Option = ggufembed.Option

// Options helpers for configuring the runtime.
var (
	WithThreads   = ggufembed.WithThreads
	WithBatchSize = ggufembed.WithBatchSize
	WithVerbose   = ggufembed.WithVerbose
)

// Input is a task-aware embedding request.
type Input = ggufembed.EmbedInput

// Runtime wraps the underlying embedding runtime and exposes a simplified API.
type Runtime struct {
	inner ggufembed.Runtime
}

// Open loads a GGUF model from disk and returns a Runtime.
func Open(path string, opts ...Option) (*Runtime, error) {
	rt, err := ggufembed.Open(path, opts...)
	if err != nil {
		return nil, err
	}
	return &Runtime{inner: rt}, nil
}

// OpenBytes loads a GGUF model directly from an in-memory byte slice.
func OpenBytes(data []byte, opts ...Option) (*Runtime, error) {
	rt, err := ggufembed.OpenBytes(data, opts...)
	if err != nil {
		return nil, err
	}
	return &Runtime{inner: rt}, nil
}

// Close releases resources associated with the runtime.
func (r *Runtime) Close() error {
	return r.inner.Close()
}

// EmbedDim reports the native embedding dimensionality of the loaded model.
func (r *Runtime) EmbedDim() int {
	return r.inner.EmbedDim()
}

// MaxSeqLen reports the maximum supported input sequence length.
func (r *Runtime) MaxSeqLen() int {
	return r.inner.MaxSeqLen()
}

// Embed embeds a batch of texts using the provided task and dimensionality.
func (r *Runtime) Embed(ctx context.Context, texts []string, task Task, dim Dimensions) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	resolvedDim, err := resolveDimensions(dim)
	if err != nil {
		return nil, err
	}
	inputs := make([]ggufembed.EmbedInput, len(texts))
	for i, text := range texts {
		inputs[i] = ggufembed.EmbedInput{
			Task:    task,
			Content: text,
			Dim:     resolvedDim,
		}
	}
	return r.inner.EmbedInputs(ctx, inputs)
}

// EmbedInputs embeds task-aware inputs (advanced usage).
func (r *Runtime) EmbedInputs(ctx context.Context, inputs []Input) ([][]float32, error) {
	return r.inner.EmbedInputs(ctx, inputs)
}

// EmbedSingle embeds a single text using the provided task and dimensionality.
func (r *Runtime) EmbedSingle(ctx context.Context, text string, task Task, dim Dimensions) ([]float32, error) {
	results, err := r.Embed(ctx, []string{text}, task, dim)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("embed: runtime returned no embeddings")
	}
	return results[0], nil
}

// EmbedSingleInput embeds a single task-aware input (advanced usage).
func (r *Runtime) EmbedSingleInput(ctx context.Context, input Input) ([]float32, error) {
	return r.inner.EmbedSingleInput(ctx, input)
}

// Inner exposes the underlying runtime for advanced integrations.
func (r *Runtime) Inner() ggufembed.Runtime {
	return r.inner
}

// ResolveDim re-exports the lower-level dimension resolver for compatibility.
func ResolveDim(dim int) (int, error) {
	return ggufembed.ResolveDim(dim)
}

// SupportedDims re-exports the supported dimension list.
func SupportedDims() []int {
	return ggufembed.SupportedDims()
}

func resolveDimensions(dim Dimensions) (int, error) {
	if dim == DimensionsAuto {
		return ggufembed.ResolveDim(0)
	}
	return ggufembed.ResolveDim(int(dim))
}
