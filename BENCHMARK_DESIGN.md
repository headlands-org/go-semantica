# Benchmark Design: Minimal Coverage Matrix

## Goal
Define the minimum set of benchmark scenarios that capture all real-world use cases for embedding generation, with stable measurements suitable for llama.cpp comparison.

## Benchmark Scenarios

### 1. Idle Memory
**Use Case**: Understanding base memory footprint
- **Metric**: Heap memory after model load (MB)
- **Method**: Load model, force GC, measure HeapAlloc
- **Llama.cpp equivalent**: RSS after model load, no inference

### 2. Single Short Document (Warm)
**Use Case**: Interactive applications, low-latency requirements
- **Document**: "The quick brown fox jumps over the lazy dog" (9 words)
- **Metrics**: P50, P95, P99 latency (ms) over 20 runs
- **Method**: Warmup 5 runs, measure 20 runs
- **Llama.cpp equivalent**: Same document, 20 runs with timing

### 3. Single Long Document (Warm)
**Use Case**: Full-length documents, articles, passages
- **Document**: 50-word ML passage (from current latency test)
- **Metrics**: P50, P95, P99 latency (ms) over 20 runs
- **Method**: Warmup 5 runs, measure 20 runs
- **Llama.cpp equivalent**: Same document, 20 runs with timing

### 4. Batch Throughput (Short Documents)
**Use Case**: High-volume processing, batch jobs
- **Documents**: 5 varied short sentences (6-9 words each)
- **Batch Size**: 96 texts
- **Duration**: 20 seconds continuous
- **Metrics**:
  - Throughput (embeddings/sec)
  - Peak memory (MB)
  - Avg latency per embedding (ms)
- **Method**: Continuous batching, random selection from 5 docs
- **Llama.cpp equivalent**: Same 5 docs, same batch size, 20 sec run

### 5. Batch Throughput (Long Documents)
**Use Case**: Processing full articles/passages in batch
- **Documents**: Same 50-word ML passage repeated
- **Batch Size**: 96 texts
- **Duration**: 20 seconds continuous
- **Metrics**:
  - Throughput (embeddings/sec)
  - Peak memory (MB)
  - Avg latency per embedding (ms)
- **Method**: Continuous batching, all same long doc
- **Llama.cpp equivalent**: Same doc, same batch size, 20 sec run

## Why These 5 Scenarios?

1. **Idle Memory**: Baseline resource cost
2. **Single Short**: Minimum latency achievable (interactive use)
3. **Single Long**: Realistic document latency
4. **Batch Short**: Maximum throughput capability
5. **Batch Long**: Realistic batch processing performance

These 5 rows capture:
- Memory usage (idle + peak)
- Latency (short + long documents)
- Throughput (short + long documents)
- Single vs batch processing

## Test Documents (Exact Text for Reproducibility)

### Short Document (9 words)
```
The quick brown fox jumps over the lazy dog
```

### Long Document (49 words)
```
Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.
```

### Short Document Set (5 varied, 6-9 words each)
```
1. The quick brown fox jumps over the lazy dog
2. Artificial intelligence is transforming modern technology
3. Machine learning enables computers to learn from data
4. Neural networks process information efficiently
5. Deep learning powers many AI applications
```

## Output Format

### Human-Readable (Default)
```
=== Benchmark Results ===

Platform: AMD Ryzen 9 7900 12-Core, 24 cores, linux/amd64

Scenario                        Metric              Value       Unit
------------------------------------------------------------------------
Idle Memory                     Heap Allocated      54          MB

Single Short Doc (9w)           P50 Latency         18.2        ms
                                P95 Latency         19.8        ms
                                P99 Latency         21.3        ms

Single Long Doc (49w)           P50 Latency         740.0       ms
                                P95 Latency         746.0       ms
                                P99 Latency         750.0       ms

Batch Short Docs (96x)          Throughput          83.4        emb/sec
                                Peak Memory         133         MB
                                Avg Latency         12.0        ms/emb

Batch Long Docs (96x)           Throughput          13.0        emb/sec
                                Peak Memory         140         MB
                                Avg Latency         76.9        ms/emb
```

### JSON Output (for comparisons)
```json
{
  "platform": {
    "cpu": "AMD Ryzen 9 7900 12-Core Processor",
    "cores": 24,
    "os": "linux",
    "arch": "amd64"
  },
  "timestamp": "2025-10-27T11:00:00Z",
  "model": "embeddinggemma-300m-Q8_0.gguf",
  "model_size_mb": 314,
  "scenarios": {
    "idle_memory": {
      "heap_mb": 54
    },
    "single_short": {
      "document_words": 9,
      "p50_ms": 18.2,
      "p95_ms": 19.8,
      "p99_ms": 21.3
    },
    "single_long": {
      "document_words": 49,
      "p50_ms": 740.0,
      "p95_ms": 746.0,
      "p99_ms": 750.0
    },
    "batch_short": {
      "batch_size": 96,
      "document_words_avg": 7.6,
      "throughput_emb_sec": 83.4,
      "peak_memory_mb": 133,
      "avg_latency_ms": 12.0
    },
    "batch_long": {
      "batch_size": 96,
      "document_words": 49,
      "throughput_emb_sec": 13.0,
      "peak_memory_mb": 140,
      "avg_latency_ms": 76.9
    }
  }
}
```

## Llama.cpp Comparison Script

### Benchmark Runner
```bash
#!/bin/bash
# benchmark_llamacpp.sh

MODEL="model/embeddinggemma-300m-Q8_0.gguf"
OUTPUT="results_llamacpp.json"

# Test documents (must match exactly)
SHORT_DOC="The quick brown fox jumps over the lazy dog"
LONG_DOC="Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."

# Run benchmarks
# ... implementation details ...

# Output JSON matching our format
```

### Comparison Tool
```bash
#!/bin/bash
# compare.sh

# Run pure-go-llamas
./gemma-benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive -json > results_go.json

# Run llama.cpp
./benchmark_llamacpp.sh > results_llamacpp.json

# Compare
./compare_results.py results_go.json results_llamacpp.json
```

## Stability Requirements

For each metric to be considered stable:
1. **Latency tests**: Coefficient of variation (stddev/mean) < 5%
2. **Throughput tests**: Run duration ≥20 seconds, variance < 10%
3. **Memory tests**: Force GC, wait 100ms, sample 3× and average

## Implementation Plan

### Phase 1: Update gemma-benchmark tool
1. Add `-json` flag for JSON output
2. Add 5 specific scenarios to comprehensive mode
3. Use exact test documents defined above
4. Output both human-readable and JSON

### Phase 2: Create llama.cpp benchmark script
1. Write shell script that runs llama.cpp with same docs
2. Parse timing output
3. Generate matching JSON

### Phase 3: Comparison tooling
1. Python script to diff two JSON files
2. Generate comparison table
3. Calculate % differences

### Phase 4: Documentation
1. Update README with benchmark results
2. Document how to run comparisons
3. Add interpretation guide
