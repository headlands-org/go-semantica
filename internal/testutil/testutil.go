// Package testutil provides shared helpers for tests.
package testutil

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ModelEnvVar is the environment variable used to override the model path.
const ModelEnvVar = "GO_SEMANTICA_MODEL"

// RequireModelPath returns a path to a GGUF model suitable for tests that
// need a real tokenizer/runtime. If GO_SEMANTICA_MODEL is set, that path is
// used; otherwise it falls back to the model bundled at model/embeddinggemma-300m-Q8_0.gguf.
// If no model is available, the test is skipped.
func RequireModelPath(t testing.TB) string {
	t.Helper()

	if p := os.Getenv(ModelEnvVar); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("model from %s=%q not accessible: %v", ModelEnvVar, p, err)
		}
		return p
	}

	p := defaultModelPath()
	if p == "" {
		t.Skipf("no model available; set %s to a GGUF file to run this test", ModelEnvVar)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("default model not found at %s: %v (set %s to override)", p, err, ModelEnvVar)
	}
	return p
}

// defaultModelPath returns the bundled model path resolved relative to this
// source file, so it works regardless of the test's working directory.
func defaultModelPath() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// file = <repo>/internal/testutil/testutil.go
	repoRoot := filepath.Join(filepath.Dir(file), "..", "..")
	return filepath.Clean(filepath.Join(repoRoot, "model", "embeddinggemma-300m-Q8_0.gguf"))
}
