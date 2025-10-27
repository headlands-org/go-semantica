# Performance Optimization Plan: Close the Gap with llama.cpp

**Goal**: Reduce the 23-41× performance gap through single-threaded optimizations inspired by llama.cpp's implementation.

**Constraints**:
- Must remain single-threaded (no fine-grained parallelism)
- Focus on low-hanging fruit optimizations
- Maintain correctness (all tests must pass)

**Target**: Achieve 5-10× speedup (bringing us to 4-8× slower than llama.cpp, down from 23-41×)

---

## Step 1: Optimize AVX2 SIMD Kernels

**Rationale**: Our current implementation processes 16 bytes/iteration when we could do 32 bytes using full YMM registers. llama.cpp uses `VPMADDUBSW` (multiply unsigned×signed bytes) which is more efficient than our current `VPMOVSXBW` (sign-extend) approach.

### Task 1.1: Implement Optimized AVX2 Dot Product

**Goal**: Replace current 16-byte INT8 dot product with 32-byte version using `VPMADDUBSW` instruction pattern.

**Success Criteria**:
- [ ] New assembly function `dotProductINT8AVX2Optimized` processes 32 bytes per iteration
- [ ] Uses `VPSIGNB` for abs/sign operations (2-cycle) instead of `VPMOVSXBW` (3-cycle)
- [ ] Uses `VPMADDUBSW` for unsigned×signed multiply-add to int16
- [ ] Uses `VPMADDWD` for horizontal add int16 pairs to int32
- [ ] Horizontal reduction at end using optimized hsum pattern
- [ ] Performance: 2× throughput improvement on large vectors (measured via benchmark)
- [ ] All existing tests pass with new implementation

**Sandbox**:
- `internal/kernels/simd_amd64.go` - Add new function
- `internal/kernels/simd_amd64.s` - Implement assembly
- `internal/kernels/simd_test.go` - Add benchmarks

**Do Not Touch**:
- `internal/kernels/simd_generic.go` - Keep fallback unchanged
- `internal/kernels/quantize.go` - Don't modify matmul logic yet (Step 2)
- Any files outside `internal/kernels/`

---

## Step 2: Implement Fused Online Softmax Attention

**Rationale**: Our current attention materializes the full QK matrix (seqLen² × nHeads × 4 bytes) and runs softmax in a separate pass. llama.cpp uses online softmax that computes QK·softmax·V in a single fused pass, eliminating the scratch buffer and improving cache locality.

### Task 2.1: Add Online Softmax Attention Kernel

**Goal**: Implement fused attention kernel that eliminates the seqLen² scratch buffer and computes attention in a single pass.

**Success Criteria**:
- [ ] New function `MultiHeadAttentionOnline` in `attention.go`
- [ ] Implements online softmax algorithm (running max M, running sum S)
- [ ] No scratch buffer allocation (was `seqLen * seqLen * nHeads` floats)
- [ ] Single pass over K/V cache (not separate QK then V passes)
- [ ] Vectorized Q·K dot products using existing `dotProductF32SIMD`
- [ ] Vectorized weighted value accumulation (V += weight * v)
- [ ] Output numerically equivalent to existing implementation (< 0.001 diff)
- [ ] Performance: 2-3× faster than current implementation (measured via benchmark)
- [ ] All attention tests pass

**Sandbox**:
- `internal/kernels/attention.go` - Add new function
- `internal/kernels/attention_test.go` - Add tests

**Do Not Touch**:
- `internal/runtime/model_int8.go` - Don't integrate yet (Step 4)
- SIMD assembly files - Use existing vectorized helpers

---

### Task 2.2: Add Vectorized MAD (Multiply-Add) Helper

**Goal**: Add `vecMadF32SIMD` helper for efficient `y += x * scalar` operations needed by online softmax.

**Success Criteria**:
- [ ] New function `vecMadF32SIMD(dst, x []float32, scalar float32, n int)` computes `dst[i] += x[i] * scalar`
- [ ] AVX2 implementation processes 32 floats per iteration (8 floats × 4 accumulators)
- [ ] Uses `VFMADD231PS` (FMA instruction)
- [ ] Scalar broadcast once before loop
- [ ] Generic fallback implementation
- [ ] Performance: 4-8× faster than scalar loop (measured via benchmark)
- [ ] Tests verify correctness against scalar implementation

**Sandbox**:
- `internal/kernels/simd_amd64.go` - Add AVX2 implementation
- `internal/kernels/simd_amd64.s` - Assembly
- `internal/kernels/simd_generic.go` - Fallback
- `internal/kernels/simd_test.go` - Tests/benchmarks

**Do Not Touch**:
- Files outside `internal/kernels/`

---

## Step 3: Add Operation Fusion

**Rationale**: GeGLU is currently computed as separate GELU activation + element-wise multiply, requiring two passes over data and an intermediate write. Fusing into a single kernel reduces memory bandwidth.

### Task 3.1: Implement Fused GeGLU Kernel

**Goal**: Replace separate GELU + multiply with single fused kernel.

**Success Criteria**:
- [ ] New function `GeGLUFused(output, gate, up []float32, n int)` computes `output[i] = GELU(gate[i]) * up[i]`
- [ ] Single pass over data (no intermediate buffer)
- [ ] Vectorized with AVX2 (process 8 floats per iteration)
- [ ] Uses quick GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
- [ ] Performance: 10-15% faster than separate ops (measured via benchmark)
- [ ] Output matches separate GELU+multiply within float32 precision
- [ ] Tests verify correctness

**Sandbox**:
- `internal/kernels/activation.go` - Add new function
- `internal/kernels/activation_test.go` - Add tests/benchmarks

**Do Not Touch**:
- Existing `GELUQuick` function - Keep for compatibility
- `internal/runtime/model_int8.go` - Don't integrate yet (Step 4)

---

## Step 4: Optimize Memory Allocation

**Rationale**: Currently allocating 28+ buffers per forward pass (hidden, residual, q, k, v, etc. × layers) causes GC pressure. Single arena allocation + buffer reuse reduces allocations to 1 per forward pass.

### Task 4.1: Implement Buffer Arena Allocator

**Goal**: Single large buffer pool that gets sliced for different uses, eliminating per-layer allocations.

**Success Criteria**:
- [ ] New `BufferArena` struct in `internal/runtime/buffers.go`
- [ ] Method `NewArena(config, seqLen) *BufferArena` allocates single large buffer
- [ ] Methods to get slices: `GetHidden()`, `GetQKV()`, `GetAttnScratch()`, `GetMLP()`
- [ ] Buffer size calculation accounts for all layer needs
- [ ] Arena reusable across forward passes (just reset offsets)
- [ ] Zero additional allocations in hot path after first call
- [ ] Tests verify buffer isolation (no overlap)
- [ ] Benchmarks show reduced GC overhead

**Sandbox**:
- `internal/runtime/buffers.go` - New file
- `internal/runtime/buffers_test.go` - New file

**Do Not Touch**:
- `internal/runtime/model_int8.go` - Don't integrate yet (Task 4.2)
- Other runtime files

---

### Task 4.2: Integrate Arena into Forward Pass

**Goal**: Replace all `make([]float32, ...)` calls in forward pass with arena slices.

**Success Criteria**:
- [ ] `ForwardINT8` accepts or creates arena
- [ ] All buffer allocations (hidden, residual, q, k, v, attnScratch, etc.) use arena slices
- [ ] Attention scratch buffer reused across layers
- [ ] Single allocation per forward pass (the arena itself)
- [ ] All existing tests pass
- [ ] Performance: 10-15% improvement from reduced GC (measured via benchmark)

**Sandbox**:
- `internal/runtime/model_int8.go` - Modify `ForwardINT8` and layer functions
- `internal/runtime/model.go` - Update `Forward` to use arena

**Do Not Touch**:
- Public API (`pkg/ggufembed/`) - Keep interface unchanged
- Kernel functions - They accept slices, don't care about allocation

---

## Step 5: Integration and Measurement

**Rationale**: Integrate all optimizations and measure cumulative impact.

### Task 5.1: Wire Up Optimized Kernels

**Goal**: Replace old kernel calls with new optimized versions throughout the codebase.

**Success Criteria**:
- [ ] `matMulQ8_0INT8Serial` calls `dotProductINT8AVX2Optimized`
- [ ] `runAttentionINT8` calls `MultiHeadAttentionOnline`
- [ ] `runMLPINT8` calls `GeGLUFused`
- [ ] All runtime tests pass
- [ ] All pkg/ggufembed tests pass
- [ ] Benchmark shows cumulative improvement

**Sandbox**:
- `internal/kernels/quantize.go` - Update dot product call
- `internal/runtime/model_int8.go` - Update attention and MLP calls

**Do Not Touch**:
- Public API signatures
- Test files (unless they need new fixtures)

---

### Task 5.2: Performance Validation

**Goal**: Measure and document performance improvement.

**Success Criteria**:
- [ ] Run `profile_single_long.go` and analyze CPU profile
- [ ] Update `PROFILING_ANALYSIS.txt` with new timings
- [ ] Run comprehensive benchmark: `./benchmark -mode=comprehensive`
- [ ] Verify improvement: target 5-10× speedup (138ms → 15-30ms for short, 735ms → 80-150ms for long)
- [ ] Run llama.cpp comparison: `./scripts/benchmark_llamacpp.sh`
- [ ] Update README.md with new performance numbers
- [ ] All tests pass

**Sandbox**:
- `PROFILING_ANALYSIS.txt` - Update with new analysis
- `README.md` - Update benchmark table
- New files for analysis/documentation

**Do Not Touch**:
- Source code (no changes, just measurement)

---

## Step 6: Polish and Documentation

### Task 6.1: Code Cleanup

**Goal**: Remove unused code, add comments, ensure consistency.

**Success Criteria**:
- [ ] Add performance-oriented comments to new kernels
- [ ] Document why we chose specific block sizes
- [ ] Update CLAUDE.md with new optimization insights
- [ ] Remove any dead code from refactoring
- [ ] Consistent code style

**Sandbox**:
- Any file that was modified in previous steps
- `CLAUDE.md` - Documentation updates

**Do Not Touch**:
- Functional logic (no behavior changes)

---

## Expected Cumulative Impact

| Optimization | Expected Improvement | Cumulative |
|--------------|---------------------|------------|
| Baseline | 138ms (short), 735ms (long) | - |
| AVX2 dot product (32 bytes) | 2× on matmul | 69ms, 370ms |
| Online softmax attention | 3× on attention | 55ms, 250ms |
| GeGLU fusion | 1.1× on FFN | 50ms, 230ms |
| Arena allocation | 1.1× overall | 45ms, 210ms |
| **Total** | **3.1× faster (short), 3.5× faster (long)** | **138→45ms, 735→210ms** |

**Gap reduction**: 23× → 7.5× (short), 41× → 12× (long)

This brings us much closer to llama.cpp while maintaining pure-Go simplicity and single-threaded execution.
