#!/bin/bash
# Quick validation script to check correctness after optimizations
# Runs the llama.cpp comparison test and checks cosine similarity

set -e

echo "=== Running Correctness Validation ==="
echo

# Run the integration test
go test -v -tags=integration ./internal/runtime -run TestEmbeddingGemmaVsLlamaCpp 2>&1 | tee /tmp/validation_output.txt

# Extract cosine similarity
COS_SIM=$(grep "Cosine similarity:" /tmp/validation_output.txt | awk '{print $3}')

echo
echo "=== Validation Result ==="
echo "Cosine similarity: $COS_SIM"

# Check if it passes threshold
if (( $(echo "$COS_SIM >= 0.98" | bc -l) )); then
    echo "✅ PASS: Cosine similarity >= 0.98"
    exit 0
else
    echo "❌ FAIL: Cosine similarity < 0.98"
    exit 1
fi
