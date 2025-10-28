// Package model provides access to the embedded GGUF model file.
//
// Import this package to use the embedded model without external files:
//
//	import "github.com/lth/pure-go-llamas/model"
//
//	rt, err := model.Open()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer rt.Close()
//
// The model file is embedded at compile time, making binaries self-contained
// but large (~300MB for Gemma-300M).
package model

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

//go:embed embeddinggemma-300m-Q8_0.gguf
var embeddedModelBytes []byte

// Open loads the embedded model and returns a Runtime.
//
// The embedded model is written to a temporary file (required for mmap support)
// which is automatically cleaned up when the runtime is closed.
//
// Options can be passed to configure the runtime (e.g., WithThreads, WithVerbose).
func Open(opts ...ggufembed.Option) (ggufembed.Runtime, error) {
	// Write embedded model to temporary file
	tmpFile, err := os.CreateTemp("", "embedded-model-*.gguf")
	if err != nil {
		return nil, fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()

	if _, err := tmpFile.Write(embeddedModelBytes); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return nil, fmt.Errorf("write embedded model: %w", err)
	}
	tmpFile.Close()

	// Load model from temporary file
	rt, err := ggufembed.Open(tmpPath, opts...)
	if err != nil {
		os.Remove(tmpPath)
		return nil, fmt.Errorf("load embedded model: %w", err)
	}

	// Wrap to clean up temp file on close
	return &embeddedRuntime{
		Runtime:      rt,
		tempFilePath: tmpPath,
	}, nil
}

// embeddedRuntime wraps ggufembed.Runtime to clean up temp file on close
type embeddedRuntime struct {
	ggufembed.Runtime
	tempFilePath string
}

// Close releases resources and removes the temporary model file
func (r *embeddedRuntime) Close() error {
	err := r.Runtime.Close()
	if removeErr := os.Remove(r.tempFilePath); removeErr != nil && err == nil {
		err = removeErr
	}
	return err
}
