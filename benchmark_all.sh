#!/bin/bash
# Comprehensive benchmark script

set -e

echo "=== Pure-Go-Llamas Performance Benchmark ==="
echo "Model: EmbeddingGemma-300M Q8_0"
echo "Date: $(date)"
echo "CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"
echo

echo "=== Test 1: Similarity Example (3 texts) ==="
go run ./examples/similarity ./model/embeddinggemma-300m-Q8_0.gguf 2>&1 | grep -E "(Generated|Similarity scores|Most similar)" | head -10
echo

echo "=== Test 2: Batch Processing (100 texts) ==="
go run ./examples/batch ./model/embeddinggemma-300m-Q8_0.gguf 2>&1 | grep -E "(Batch embedding|Average time|Throughput)"
echo

echo "=== Test 3: Correctness Validation ==="
go test -v -tags=integration ./internal/runtime -run TestEmbeddingGemmaVsLlamaCpp 2>&1 | grep -E "(Cosine similarity|Max difference|RMS difference)"
echo

echo "=== Benchmark Complete ==="
