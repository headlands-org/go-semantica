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

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

//go:embed embeddinggemma-300m-Q8_0.gguf
var embeddedModelBytes []byte

// Open loads the embedded model and returns a Runtime directly from memory.
//
// Options can be passed to configure the runtime (e.g., WithThreads, WithVerbose).
func Open(opts ...ggufembed.Option) (ggufembed.Runtime, error) {
	return ggufembed.OpenBytes(embeddedModelBytes, opts...)
}
