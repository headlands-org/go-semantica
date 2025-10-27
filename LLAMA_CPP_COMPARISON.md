# Llama.cpp Comparison Methodology

## Goal
Provide apples-to-apples performance comparison between pure-go-llamas and llama.cpp for embedding generation using the same model.

## Test Setup

### Model
- **Model**: `embeddinggemma-300M-Q8_0.gguf`
- **Location**: Same file used for both implementations
- **Format**: GGUF v3, Q8_0 quantization
- **Size**: ~314 MB

### Test Document
Use identical test document for both:
```
"Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."
```
(~50 words, representative of typical embedding use cases)

### Metrics to Compare

1. **Idle Memory**: Model loaded, no inference (measure via RSS/HeapAlloc)
2. **Single Document Latency** (warm): P50/P95/P99 of 20 runs
3. **Maximum Throughput**: Embeddings/sec over 20 second window
4. **Peak Memory**: Memory usage during throughput test

## Llama.cpp Benchmark Script

### Prerequisites
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Verify embedding support
./embedding --help
```

### Test 1: Idle Memory
```bash
# Start llama.cpp with model loaded, measure RSS before any inference
/usr/bin/time -v ./embedding -m ../pure-go-llamas/model/embeddinggemma-300m-Q8_0.gguf --n-predict 0 2>&1 | grep "Maximum resident set size"
```

### Test 2: Single Document Latency
```bash
# Create test document
cat > test_doc.txt << 'EOF'
Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.
EOF

# Run 20 times and calculate percentiles
for i in {1..20}; do
  /usr/bin/time -f "%e" ./embedding -m ../pure-go-llamas/model/embeddinggemma-300m-Q8_0.gguf -p "$(cat test_doc.txt)" 2>&1 | tail -1
done | sort -n | awk '
  {times[NR]=$1}
  END {
    print "Mean:", (times[1]+times[NR])/2
    print "P50:", times[int(NR*0.5)]
    print "P95:", times[int(NR*0.95)]
    print "P99:", times[int(NR*0.99)]
  }
'
```

### Test 3: Maximum Throughput
```bash
# Create batch of documents (1000 copies)
for i in {1..1000}; do cat test_doc.txt; done > batch_docs.txt

# Measure throughput with time
START=$(date +%s.%N)
./embedding -m ../pure-go-llamas/model/embeddinggemma-300m-Q8_0.gguf -p "$(cat batch_docs.txt)" -n 20 2>&1
END=$(date +%s.%N)
DURATION=$(echo "$END - $START" | bc)
THROUGHPUT=$(echo "1000 / $DURATION" | bc -l)
echo "Throughput: $THROUGHPUT embeddings/sec"
```

### Test 4: Peak Memory During Throughput
```bash
# Run throughput test with memory monitoring
./embedding -m ../pure-go-llamas/model/embeddinggemma-300m-Q8_0.gguf -p "$(cat batch_docs.txt)" -n 20 &
PID=$!

# Sample memory every 100ms
MAX_MEM=0
while kill -0 $PID 2>/dev/null; do
  MEM=$(ps -p $PID -o rss= | awk '{print $1}')
  if [ $MEM -gt $MAX_MEM ]; then
    MAX_MEM=$MEM
  fi
  sleep 0.1
done

echo "Peak Memory: $((MAX_MEM / 1024)) MB"
```

## Running the Comparison

### Step 1: Run pure-go-llamas benchmark
```bash
cd pure-go-llamas
./gemma-benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive > results_go.txt
```

### Step 2: Run llama.cpp benchmarks
```bash
cd llama.cpp
./benchmark_embedding.sh > results_llamacpp.txt
```

### Step 3: Compare results
```bash
# Generate comparison table
./compare_results.sh results_go.txt results_llamacpp.txt
```

## Expected Output Format

```
=== Performance Comparison: pure-go-llamas vs llama.cpp ===

Platform: AMD Ryzen 9 7900 12-Core (24 threads), Linux

| Metric                     | pure-go-llamas | llama.cpp | Difference |
|----------------------------|----------------|-----------|------------|
| Idle Memory (MB)           | 54             | ~300      | -82%       |
| Single Doc Latency P50 (ms)| 18.5          | 17.2      | +7.6%      |
| Single Doc Latency P95 (ms)| 19.8          | 18.9      | +4.8%      |
| Max Throughput (emb/sec)   | 55.3           | 62.1      | -11%       |
| Peak Memory (MB)           | 133            | ~450      | -70%       |

Notes:
- Both tests use identical model (embeddinggemma-300M-Q8_0.gguf)
- Both tests use identical test document
- pure-go-llamas uses coarse-grained parallelism (48 workers)
- llama.cpp uses default threading configuration
```

## Alternative: Integration Test

Instead of manual scripts, we could create an integration test that:
1. Checks if llama.cpp is installed (`which embedding`)
2. Runs both benchmarks automatically
3. Compares results side-by-side
4. Optionally uploads to a comparison dashboard

### Implementation
```go
// cmd/compare-llamacpp/main.go
func main() {
    // Check llama.cpp availability
    // Run pure-go-llamas benchmark
    // Run llama.cpp benchmark
    // Compare and report
}
```

## Notes

### Why This Comparison is Fair
- Same model file (byte-for-byte identical)
- Same test documents
- Same hardware
- Same measurement methodology
- Both use Q8_0 quantization

### Known Differences
- **Memory**: pure-go-llamas uses memory-mapped I/O (zero-copy), llama.cpp loads into memory
- **Threading**: Different parallelism strategies (coarse vs fine-grained)
- **SIMD**: Both use CPU-specific optimizations (AVX2 on x86_64, NEON on ARM)

### Caveats
- Llama.cpp is highly optimized C++ with extensive SIMD
- pure-go-llamas prioritizes portability and simplicity
- Performance gap expected, but should be <2x for INT8 inference
- Memory advantage should favor pure-go-llamas significantly
