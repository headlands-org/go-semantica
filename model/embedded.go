// Package model provides access to the embedded GGUF model file.
//
// Import this package to use the embedded model without external files:
//
//	import "github.com/headlands-org/go-semantica/model"
//
//	rt := model.MustOpen()
//	defer rt.Close()
//
// The model file is embedded at compile time, making binaries self-contained
// but large (~300MB for Gemma-300M).
package model

import (
	_ "embed"

	"github.com/headlands-org/go-semantica"
)

//go:embed embeddinggemma-300m-Q8_0.gguf
var embeddedModelBytes []byte

// Open loads the embedded model and returns a Runtime directly from memory.
//
// Options can be passed to configure the runtime (e.g., WithThreads, WithVerbose).
func Open(opts ...semantica.Option) (*semantica.Runtime, error) {
	return semantica.OpenBytes(embeddedModelBytes, opts...)
}

// MustOpen is like Open but panics if the embedded model cannot be loaded.
func MustOpen(opts ...semantica.Option) *semantica.Runtime {
	rt, err := Open(opts...)
	if err != nil {
		panic(err)
	}
	return rt
}
