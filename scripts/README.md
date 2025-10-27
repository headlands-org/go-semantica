# Benchmark Scripts

Scripts for running and comparing embedding benchmarks.

## Quick Start

### Run pure-go-llamas benchmark
```bash
./benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive
```

### Run llama.cpp benchmark (requires llama.cpp installation)
```bash
./scripts/benchmark_llamacpp.sh model/embeddinggemma-300m-Q8_0.gguf ../llama.cpp
```

## Installing llama.cpp

```bash
# Clone llama.cpp (sibling to pure-go-llamas)
cd ..
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make -j

# Verify
./embedding --help

# Return to pure-go-llamas
cd ../pure-go-llamas
```

## Running Comparisons

### 1. Run both benchmarks
```bash
# Run pure-go-llamas (save output)
./benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive > results_go.txt

# Run llama.cpp (save output)
./scripts/benchmark_llamacpp.sh model/embeddinggemma-300m-Q8_0.gguf ../llama.cpp > results_llamacpp.txt
```

### 2. Compare side-by-side
```bash
# View both results
echo "=== pure-go-llamas ===" && cat results_go.txt
echo ""
echo "=== llama.cpp ===" && cat results_llamacpp.txt
```

## Benchmark Scenarios

Both benchmarks test these scenarios:

1. **Idle Memory**: Memory footprint with model loaded
2. **Single Short Doc (9w)**: Latency for 9-word document
3. **Single Long Doc (49w)**: Latency for 49-word document
4. **Batch Short Docs (96×)**: Throughput for short documents
5. **Batch Long Docs (96×)**: Throughput for long documents

### Test Documents

**Short (9 words):**
```
The quick brown fox jumps over the lazy dog
```

**Long (49 words):**
```
Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.
```

## Known Limitations

### Llama.cpp Batch Testing
The `llama.cpp/embedding` tool doesn't expose a batch API, so scenarios 4 & 5 (batch throughput) cannot be directly compared. To properly benchmark llama.cpp's batch performance, you would need to:

1. Use llama.cpp's C++ API directly
2. Write a custom harness that calls the embedding API in a loop
3. Manage parallelism similar to our batch mode

For now, the script only measures single-document latency (scenarios 1-3).

## Interpreting Results

### What We Measure

**llama.cpp**: Uses internal "prompt eval time" metric (inference only)
- Excludes model loading (~160ms overhead per process)
- Measures pure computation time for fair comparison

**pure-go-llamas**: Measures inference time with model already loaded
- Directly comparable to llama.cpp's prompt eval time

### Expected Differences

**Memory:**
- pure-go-llamas uses memory-mapped I/O (zero-copy)
- llama.cpp loads model into memory
- Expect pure-go-llamas to use 5-10× less memory

**Latency:**
- llama.cpp is highly optimized C++ with extensive SIMD
- Current results: llama.cpp 20-40× faster on single documents
- pure-go-llamas prioritizes portability and simplicity

**Throughput:**
- Both saturate CPU effectively
- pure-go-llamas uses coarse-grained parallelism (worker pools)
- llama.cpp uses fine-grained parallelism within operations

### Fair Comparison Criteria

✅ **Same model file** (byte-for-byte identical)
✅ **Same test documents** (exact same text)
✅ **Same hardware** (run on same machine)
✅ **Same quantization** (Q8_0 INT8)
✅ **Warm cache** (both use warmup runs)

## Troubleshooting

### llama.cpp not found
```
Error: llama.cpp embedding binary not found
```
**Solution**: Install llama.cpp following instructions above.

### Model not found
```
Error: Model not found at model/embeddinggemma-300m-Q8_0.gguf
```
**Solution**: Download model or update path:
```bash
# If model is elsewhere
./scripts/benchmark_llamacpp.sh /path/to/model.gguf ../llama.cpp
```

### Different results on reruns
This is normal - some variance is expected. Run each benchmark 3 times and average the results for stability.
