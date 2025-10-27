# Pure-Go-LLamas Optimization Plan
**Goal:** Close the 7.9-19.3× performance gap with llama.cpp
**Current State:** 51.1ms P50 latency (short docs) vs llama.cpp's 6.5ms
**Target:** Achieve 4-8× improvement to reach 6-13ms range

---

## Performance Gap Analysis

**Current Bottlenecks (ranked by impact):**
1. **Missing SIMD in non-matmul ops:** ~40M FLOPs/inference (84% potential reduction)
2. **Matmul inefficiencies:** Missing VNNI, prefetch, register blocking (40-50% potential gain)
3. **Dynamic quantization overhead:** 6.45% of current runtime
4. **GC pressure:** 12.28% background overhead

**Theoretical Headroom:**
- llama.cpp efficiency: 36.8% of hardware peak
- Our efficiency: 4.3% of hardware peak
- **Gap to close: 8.5× efficiency improvement**

---

## Execution Strategy

This plan follows a **sequential step approach** where each step builds on the previous. Within each step, tasks can be executed in parallel. All optimizations will be validated against the gold file (`testdata/reference_embedding_full.txt`) to ensure 0.988+ cosine similarity.

---

## Step 1: Quick Wins - Vectorize Non-MatMul Operations
**Estimated Impact:** 3-4× speedup (40M FLOPs → ~8M FLOPs)
**Risk:** Low (reusing existing SIMD infrastructure)
**Timeline:** 2-3 days

### Task 1.1: Vectorize Attention QK^T Dot Products
**Goal:** Replace scalar dot products in `MultiHeadAttention` with existing `dotProductINT8SIMD`

**Success Criteria:**
- [ ] Attention score computation uses `dotProductINT8SIMD` for all Q·K operations
- [ ] Latency reduction of 1.5-2× measured in benchmarks
- [ ] Cosine similarity vs llama.cpp ≥ 0.98

**Sandbox (allowed to modify):**
- `internal/kernels/attention.go` - `MultiHeadAttentionWithScale` function
- `internal/kernels/attention_test.go` - Add SIMD-specific tests

**Do Not Touch:**
- `internal/kernels/simd_amd64.s` - Reuse existing assembly
- `internal/runtime/` - Keep model logic unchanged
- Tokenizer, GGUF parsing

**Implementation Notes:**
- Current code has nested loops computing `score += Q[i] * K[j]`
- Replace with single call to `dotProductINT8SIMD(Q[i:i+headDim], K[j:j+headDim], headDim)`
- Handle quantization if needed (Q/K are FP32 in Gemma)

---

### Task 1.2: Vectorize Element-Wise Operations
**Goal:** Add AVX2 SIMD implementations for `VecMulF32`, `VecAddF32`, `VecScaleF32`

**Success Criteria:**
- [ ] New functions `vecMulF32SIMD`, `vecAddF32SIMD`, `vecScaleF32SIMD` in `simd_amd64.go`
- [ ] Assembly implementations in `simd_amd64.s` using `VMULPS`, `VADDPS`, `VMULPS+VBROADCASTSS`
- [ ] Performance improvement: 6-8× faster than scalar (measured via benchmarks)
- [ ] Fallback to scalar on non-AVX2 platforms

**Sandbox:**
- `internal/kernels/simd_amd64.s` - Add new assembly functions
- `internal/kernels/simd_amd64.go` - Add Go declarations
- `internal/kernels/simd_generic.go` - Add scalar fallbacks
- `internal/kernels/matmul.go` - Update callers to use SIMD versions

**Do Not Touch:**
- Quantization logic
- Model forward pass structure
- Existing matmul assembly

**Implementation Notes:**
- Process 8 floats at a time with AVX2 (256-bit registers)
- Handle remainders with scalar loop
- Similar pattern to existing `dotProductINT8SIMD`

---

### Task 1.3: Vectorize GELU Activation
**Goal:** Create SIMD-optimized `GELUQuickSIMD` using polynomial approximation

**Success Criteria:**
- [ ] New function `GELUQuickSIMD` with AVX2 implementation
- [ ] Uses polynomial approximation for sigmoid (avoid expensive `exp`)
- [ ] Performance: 2-4× faster than scalar `GELUQuick`
- [ ] Numerical accuracy: max error < 0.01 vs scalar version

**Sandbox:**
- `internal/kernels/activation.go` - Add SIMD variant
- `internal/kernels/simd_amd64.s` - Assembly implementation
- `internal/kernels/activation_test.go` - Accuracy tests

**Do Not Touch:**
- Existing scalar `GELU` and `GELUQuick` (keep for fallback)
- FFN calling code (automatic dispatch)

**Implementation Notes:**
- Sigmoid approximation: `1/(1+exp(-x)) ≈ polynomial(x)`
- Use Horner's method for polynomial evaluation
- Reference: llama.cpp's fast sigmoid implementations

---

## Step 2: Eliminate Dynamic Quantization Overhead
**Estimated Impact:** 1.1-1.15× speedup (remove 6.45% bottleneck)
**Risk:** Medium (refactoring model loading)
**Timeline:** 1-2 days

### Task 2.1: Pre-Quantize Static Weights at Load Time
**Goal:** Quantize all layer weights to INT8 during model loading, avoiding per-inference quantization

**Success Criteria:**
- [ ] Model struct stores pre-quantized INT8 weights for all layers
- [ ] `QuantizeSymmetricINT8` called only once per weight tensor during `loadLayer()`
- [ ] Runtime `Forward()` uses pre-quantized weights directly
- [ ] Memory usage increase < 10% (negligible - weights already 300MB)
- [ ] `QuantizeSymmetricINT8` removed from CPU profile hot path

**Sandbox:**
- `internal/runtime/model.go` - Modify `loadLayer()` and `Layer` struct
- `internal/runtime/model_test.go` - Update tests

**Do Not Touch:**
- Activation quantization (still dynamic, only once per layer)
- GGUF parsing logic
- Inference correctness

**Implementation Notes:**
- Change `Layer.attnQ.weight` from `[]byte` to `*QuantizedTensorINT8`
- Quantize during `parseQ8_0Block()` or immediately after weight load
- Activation quantization remains dynamic (input-dependent)

---

## Step 3: Advanced Matmul Optimizations
**Estimated Impact:** 1.5-2× speedup (stack with previous gains)
**Risk:** High (complex assembly, register management)
**Timeline:** 3-5 days

### Task 3.1: Add Software Prefetching to Matmul Loop
**Goal:** Prefetch weight and input data 2 blocks ahead in `matmulInnerLoopAsm`

**Success Criteria:**
- [ ] Assembly code includes `PREFETCHT0` instructions for next 2 blocks
- [ ] Prefetch both input row and weight block pointers
- [ ] Performance gain: 5-10% measured on single-doc benchmark
- [ ] No correctness regression

**Sandbox:**
- `internal/kernels/simd_amd64.s` - Modify `matmulInnerLoopAsm`

**Do Not Touch:**
- Core computation logic (don't break existing FMA implementation)
- Quantization logic

**Implementation Notes:**
```asm
// At top of block loop:
MOVQ    blockIdx, R10
ADDQ    $2, R10              // blockIdx + 2
IMULQ   $34, R10             // Block size = 34 bytes
PREFETCHT0 (DI)(R10*1)       // Prefetch weight[blockIdx+2]
PREFETCHT0 (SI)(R10*1)       // Prefetch input[blockIdx+2]
```

---

### Task 3.2: Implement Register Blocking (4×4 tiles)
**Goal:** Keep 4×4 output tile in YMM registers throughout matmul kernel

**Success Criteria:**
- [ ] New assembly function `matmulRegisterBlocked4x4` processes 4 output rows × 4 cols
- [ ] Uses 16 YMM registers for accumulators (4 rows × 4 cols of float32×8)
- [ ] Outer loop calls this kernel in 4×4 tiles
- [ ] Performance: 15-20% improvement over current implementation
- [ ] Handles edge cases (non-multiple-of-4 dimensions)

**Sandbox:**
- `internal/kernels/simd_amd64.s` - New assembly function
- `internal/kernels/simd_amd64.go` - Go wrapper
- `internal/kernels/quantize.go` - Integrate into `matMulQ8_0INT8Serial`

**Do Not Touch:**
- Existing `matmulInnerLoopAsm` (keep as fallback)

**Implementation Notes:**
- This is the MOST COMPLEX task - requires careful register allocation
- YMM0-YMM15: 16 accumulators for 4×4 tile
- Process multiple K iterations before writing back to memory
- Reference: llama.cpp's `gemm_bloc<4,4>()` template

---

### Task 3.3: Batch FP16→FP32 Scale Conversion
**Goal:** Convert 4 FP16 scales to FP32 in one operation using `VCVTPH2PS`

**Success Criteria:**
- [ ] Assembly code loads 4× FP16 scales (64 bits) into XMM register
- [ ] Single `VCVTPH2PS` converts to 4× FP32 in YMM register
- [ ] Performance: 3-5% improvement (small but measurable)
- [ ] Requires F16C CPU feature detection

**Sandbox:**
- `internal/kernels/simd_amd64.s` - Modify scale loading in matmul
- Add CPU feature detection (check `CPUID` for F16C support)

**Do Not Touch:**
- Fallback to scalar conversion if F16C not available

**Implementation Notes:**
```asm
MOVQ    (scalesPtr), XMM0    // Load 4× FP16 (64 bits)
VCVTPH2PS XMM0, YMM1         // Convert to 4× FP32
```

---

## Step 4: Memory & GC Optimization
**Estimated Impact:** 1.1-1.2× speedup (reduce 12% GC overhead)
**Risk:** Low (existing buffer pool infrastructure)
**Timeline:** 1 day

### Task 4.1: Expand Buffer Pooling to Hot Paths
**Goal:** Pre-allocate and reuse buffers for attention scratch space, FFN intermediates

**Success Criteria:**
- [ ] No allocations in `Forward()` hot path (verified with `go test -memprofile`)
- [ ] GC overhead reduced from 12% to < 5% in CPU profile
- [ ] Buffer pool expanded for:
  - Attention score matrices (seqLen × seqLen × nHeads)
  - FFN intermediate tensors (seqLen × intermDim)
  - Quantization buffers

**Sandbox:**
- `internal/runtime/model.go` - Expand `bufferPool` struct
- `pkg/ggufembed/runtime.go` - Pre-allocate pools per worker

**Do Not Touch:**
- Core computation kernels

**Implementation Notes:**
- Already have `sync.Pool` infrastructure
- Add pools for:
  - `attentionScoreBuffer []float32` (size: maxSeqLen² × maxHeads)
  - `ffnIntermediateBuffer []float32` (size: maxSeqLen × intermDim)
- Allocate once per worker thread, reuse across batches

---

## Step 5: Vectorize Remaining Operations
**Estimated Impact:** 1.05-1.1× speedup (diminishing returns)
**Risk:** Low
**Timeline:** 1-2 days

### Task 5.1: Vectorize RMSNorm
**Goal:** SIMD implementation of RMSNorm using AVX2

**Success Criteria:**
- [ ] New function `RMSNormSIMD` with two-pass SIMD:
  - Pass 1: Dot product for sum of squares (reuse `dotProductSIMD`)
  - Pass 2: Vectorized scale and multiply
- [ ] Performance: 4-6× faster than scalar
- [ ] Numerical accuracy maintained (max error < 1e-6)

**Sandbox:**
- `internal/kernels/norm.go`
- `internal/kernels/simd_amd64.s` (if needed for pass 2)

**Do Not Touch:**
- Existing scalar RMSNorm (fallback)

---

### Task 5.2: Vectorize RoPE
**Goal:** SIMD implementation of RoPE rotations using AVX2 FMA

**Success Criteria:**
- [ ] Process 4 element pairs at a time (8 floats total)
- [ ] Uses cached cos/sin values (already optimized)
- [ ] Performance: 4-6× faster than scalar
- [ ] Correctness validated against scalar version

**Sandbox:**
- `internal/kernels/rope_cache.go`

**Do Not Touch:**
- Cache generation logic (already efficient)

---

### Task 5.3: Vectorize Softmax
**Goal:** SIMD softmax with vectorized exp approximation

**Success Criteria:**
- [ ] AVX2 implementation using polynomial approximation for `exp(x)`
- [ ] Three passes: max reduction (SIMD), exp+sum (SIMD), normalize (SIMD)
- [ ] Performance: 2-3× faster (exp is hard to vectorize efficiently)
- [ ] Numerical stability maintained (subtract max before exp)

**Sandbox:**
- `internal/kernels/activation.go`

**Do Not Touch:**
- Keep scalar version for fallback

---

## Validation Protocol (All Steps)

**After EVERY task completion:**

1. **Run correctness validation:**
   ```bash
   ./validate_correctness.sh
   # Must show: Cosine similarity >= 0.980
   ```

2. **Run comprehensive benchmarks:**
   ```bash
   go run ./cmd/benchmark -model ./model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive
   ```

3. **Profile for regressions:**
   ```bash
   go test -bench=BenchmarkEmbeddingSingle -cpuprofile=cpu.prof ./internal/runtime
   go tool pprof -top cpu.prof
   ```

4. **Check for new allocations:**
   ```bash
   go test -bench=BenchmarkEmbeddingSingle -memprofile=mem.prof ./internal/runtime
   go tool pprof -alloc_space mem.prof
   ```

**Acceptance criteria for each task:**
- ✅ Cosine similarity ≥ 0.98 vs llama.cpp
- ✅ Performance improvement matches estimate (±20%)
- ✅ No new allocations in hot path
- ✅ All tests pass

---

## Expected Cumulative Results

| After Step | Latency (P50) | Speedup | vs llama.cpp | Efficiency |
|------------|---------------|---------|--------------|------------|
| **Baseline** | 51.1 ms | 1.0× | 7.9× slower | 4.3% |
| **Step 1** | 13-17 ms | 3-4× | 2.0-2.6× slower | 13-17% |
| **Step 2** | 11-15 ms | 1.15× | 1.7-2.3× slower | 15-19% |
| **Step 3** | 6-10 ms | 1.5-2× | 0.9-1.5× slower | 22-33% |
| **Step 4** | 5-9 ms | 1.2× | 0.8-1.4× slower | 25-37% |
| **Step 5** | 5-8 ms | 1.1× | 0.8-1.2× slower | 27-40% |

**Target:** Achieve 6-8ms P50 latency (6-8× improvement from baseline)

---

## Risk Mitigation

**High-Risk Tasks:**
- **Task 3.2 (Register blocking):** Most complex, high chance of bugs
  - Mitigation: Implement in stages (2×2 tile first, then 4×4)
  - Keep existing implementation as fallback

**Medium-Risk Tasks:**
- **Task 2.1 (Pre-quantization):** Refactoring model loading
  - Mitigation: Add extensive tests before/after
  - Validate memory usage doesn't explode

**Low-Risk Tasks:**
- All Step 1 and Step 5 tasks reuse existing patterns

---

## Dependencies

```
Step 1 (Tasks 1.1, 1.2, 1.3) → Can run in parallel
    ↓
Step 2 (Task 2.1) → Independent
    ↓
Step 3 (Tasks 3.1, 3.2, 3.3) → Sequential (3.2 builds on 3.1)
    ↓
Step 4 (Task 4.1) → Independent
    ↓
Step 5 (Tasks 5.1, 5.2, 5.3) → Can run in parallel
```

---

## Out of Scope (For This Plan)

**Not pursuing due to constraints:**
1. **VNNI instructions** - Go doesn't expose these intrinsics (would need cgo)
2. **Fine-grained parallelism** - User constraint: keep single-threaded
3. **Algorithmic changes** - Correctness is validated, focus is optimization
4. **Multi-model support** - Gemma-specific optimizations are acceptable

**Future work (after reaching 6-8ms):**
- Investigate Go→C boundary for VNNI kernels (if acceptable to user)
- Explore Go compiler improvements (PGO, escape analysis tuning)
- Support for other quantization formats (Q4_0, Q5_1)
