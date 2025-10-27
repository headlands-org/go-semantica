# Baseline Performance Results

**Date**: 2025-10-27
**Platform**: AMD Ryzen 9 7900 12-Core Processor, 24 cores, linux/amd64
**Model**: embeddinggemma-300m-Q8_0.gguf (314 MB)

## Pure-Go-Llamas Baseline

```
=== Benchmark Results ===

Platform: AMD Ryzen 9 7900 12-Core Processor, 24 cores, linux/amd64

Scenario                        Metric              Value       Unit
------------------------------------------------------------------------
Idle Memory                     Heap Allocated      54          MB

Single Short Doc (9w)           P50 Latency         140.5       ms
                                P95 Latency         150.6       ms
                                P99 Latency         150.6       ms

Single Long Doc (49w)           P50 Latency         748.5       ms
                                P95 Latency         774.2       ms
                                P99 Latency         774.2       ms

Batch Short Docs (96x)          Throughput          81.2        emb/sec
                                Peak Memory         127         MB
                                Avg Latency         12.3        ms/emb

Batch Long Docs (96x)           Throughput          12.7        emb/sec
                                Peak Memory         166         MB
                                Avg Latency         78.7        ms/emb
```

## Key Observations

### Memory Efficiency ✅
- **Idle**: Only 54 MB (excellent for embedded/serverless)
- **Peak**: 127-166 MB during batch processing
- Memory-mapped I/O provides zero-copy model access

### Performance Characteristics

**Document Length Impact**:
- Short (9w) single doc: ~140ms
- Long (49w) single doc: ~750ms
- **5.3× latency increase** for 5.4× longer document

**Batch vs Single**:
- Single short doc: 140ms
- Batch short docs: 12.3ms/emb (**11.4× faster via parallelism**)
- Single long doc: 750ms
- Batch long docs: 78.7ms/emb (**9.5× faster via parallelism**)

**Throughput**:
- Short documents: **81.2 embeddings/sec**
- Long documents: **12.7 embeddings/sec**

### Optimization Opportunities 🎯

1. **Single Document Latency**:
   - 750ms for 49-word document is slow
   - Potential for optimization (target: <100ms)

2. **Long Document Throughput**:
   - 12.7 emb/sec is limited by document length
   - Optimization could improve to 40-50 emb/sec

3. **Memory Usage**:
   - Already excellent at 54-166 MB
   - No optimization needed

## Next Steps

1. **Profile single-document path**: Identify bottlenecks causing 750ms latency
2. **Compare with llama.cpp**: Establish performance targets
3. **Optimize hot paths**: Focus on kernels that show up in profiling
4. **Re-benchmark**: Measure improvements after optimization

## Test Documents Used

**Short (9 words)**:
```
The quick brown fox jumps over the lazy dog
```

**Long (49 words)**:
```
Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing.
```

## Reproducibility

To reproduce these results:
```bash
./benchmark -model=model/embeddinggemma-300m-Q8_0.gguf -mode=comprehensive
```
