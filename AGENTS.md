# Repository Guidelines

## Project Structure & Module Organization
- Core runtime lives under `internal/`, with GGUF parsing in `internal/gguf` and execution logic in `internal/runtime`.
- Public APIs surface through the root `go-semantica` package, while the embedded model helper resides in `model/`.
- CLI tooling and samples sit in `cmd/` and `examples/`; shared test data is under `testdata/`.
- Scripts and benchmarking assets are available in `scripts/` and `benchmark_cpp/` for profiling comparisons.

## Build, Test, and Development Commands
- `go build ./...` compiles all binaries to confirm the workspace compiles cleanly.
- `go test ./...` executes unit and integration tests across packages; run before pushing.
- `go run cmd/gemma-embed` offers a quick manual sanity check using the on-disk model.
- `go run examples/similarity-embedded` exercises the embedded model path without touching the filesystem.

## Coding Style & Naming Conventions
- Follow idiomatic Go with gofmt formatting (`gofmt -w path/to/file.go`); no tabs or spaces tweaking required beyond the default.
- Favor descriptive, lowerCamelCase identifiers for locals and exported CamelCase names that match the public API.
- Keep package-level comments concise and use doc comments (`// Name ...`) for exported symbols.

## Testing Guidelines
- Tests rely on Go’s standard testing package; place new coverage in `_test.go` files alongside the code under test.
- Name tests `TestFeatureScenario` and table-driven cases `TestFeature` with subtests for clarity.
- Use fixtures from `testdata/` or package-level helpers rather than ad-hoc file writes.

## Commit & Pull Request Guidelines
- Craft imperative, present-tense commit summaries (e.g., `Improve embedding loader`); include scoped details in the body when needed.
- Associate pull requests with related issues and describe the user-visible impact, highlighting performance or API changes.
- Provide reproduction steps or sample commands for behavioral changes and attach benchmark snippets when tuning kernels or runtime paths.

## Agent Workflow Notes
- Use `go-semantica.OpenBytes` when testing embedded model flows—no temporary files should be introduced in new code.
- Prefer `rg` for searches and `apply_patch` for edits to keep diffs focused and automation-friendly.
