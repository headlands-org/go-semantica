# Examples

This directory contains focused examples demonstrating the key use cases of the pure-go-llamas embedding library.

## similarity - Semantic Similarity Comparison

A simple example showing how to:
- Generate embeddings for three sentences
- Compare them using cosine similarity
- Identify the most similar pair
- Measure timing information

The example uses two semantically similar sentences (about AI/ML) and one unrelated sentence (about food) to demonstrate how embeddings capture semantic meaning.

**Build:**
```bash
go build ./examples/similarity
```

**Run:**
```bash
./similarity model/embeddinggemma-300m-Q8_0.gguf
```

**Expected output:**
- Model load time
- Per-embedding generation time
- Pairwise similarity scores
- Identification of the most similar pair

## batch - Batch Embedding Generation

Demonstrates efficient batch processing of 100 sentences using the `Embed()` API. Shows:
- Parallel batch embedding with automatic worker pool sizing
- Throughput metrics (sentences/second)
- Average latency per sentence
- Sample embeddings from the batch

This example is ideal for understanding the performance characteristics of the library when processing many texts at once.

**Build:**
```bash
go build ./examples/batch
```

**Run:**
```bash
./batch model/embeddinggemma-300m-Q8_0.gguf
```

**Expected output:**
- Model load time
- Total batch processing time
- Throughput (sentences/second)
- Average time per sentence
- Sample embeddings with first 5 dimensions

## Performance Notes

Both examples report timing information. Typical performance on an 8-core x86 CPU:
- **Single embedding**: ~15-30ms per text (small texts)
- **Batch throughput**: ~50-70 sentences/second (100-sentence batches)
- **Model loading**: ~100-150ms (for 300M parameter model)

Performance scales with:
- Text length (longer texts = more tokens = slower)
- Batch size (larger batches = better throughput)
- CPU cores (more cores = better parallel performance)
