#!/bin/bash
set -e

echo "Building EmbeddingGemma demo..."
go build -o embeddinggemma-demo ./examples/embeddinggemma-demo/

echo ""
echo "Running demo..."
echo ""
./embeddinggemma-demo
