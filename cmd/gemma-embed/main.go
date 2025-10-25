// Command gemma-embed generates embeddings from text using GGUF models
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

var (
	modelPath  = flag.String("model", "", "Path to GGUF model file (required)")
	inputPath  = flag.String("input", "", "Input file (one text per line, default: stdin)")
	outputPath = flag.String("output", "", "Output file (default: stdout)")
	format     = flag.String("format", "json", "Output format: json, csv, tsv")
	threads    = flag.Int("threads", runtime.NumCPU(), "Number of threads")
	batchSize  = flag.Int("batch", 1, "Batch size for processing")
	verbose    = flag.Bool("verbose", false, "Verbose logging")
	showStats  = flag.Bool("stats", false, "Show performance statistics")
)

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -model is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Open model
	if *verbose {
		log.Printf("Loading model from %s...", *modelPath)
	}

	startLoad := time.Now()
	rt, err := ggufembed.Open(*modelPath,
		ggufembed.WithThreads(*threads),
		ggufembed.WithBatchSize(*batchSize),
		ggufembed.WithVerbose(*verbose),
	)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	if *verbose {
		log.Printf("Model loaded in %v", time.Since(startLoad))
		log.Printf("Embedding dimension: %d", rt.EmbedDim())
		log.Printf("Max sequence length: %d", rt.MaxSeqLen())
	}

	// Open input
	var input io.Reader = os.Stdin
	if *inputPath != "" {
		f, err := os.Open(*inputPath)
		if err != nil {
			log.Fatalf("Failed to open input file: %v", err)
		}
		defer f.Close()
		input = f
	}

	// Open output
	var output io.Writer = os.Stdout
	if *outputPath != "" {
		f, err := os.Create(*outputPath)
		if err != nil {
			log.Fatalf("Failed to create output file: %v", err)
		}
		defer f.Close()
		output = f
	}

	// Read texts
	texts := []string{}
	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			texts = append(texts, line)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("Failed to read input: %v", err)
	}

	if len(texts) == 0 {
		log.Fatalf("No input texts")
	}

	if *verbose {
		log.Printf("Processing %d texts...", len(texts))
	}

	// Generate embeddings
	startEmbed := time.Now()
	ctx := context.Background()
	embeddings, err := rt.Embed(ctx, texts)
	if err != nil {
		log.Fatalf("Failed to generate embeddings: %v", err)
	}
	embedDuration := time.Since(startEmbed)

	// Write output
	if err := writeOutput(output, *format, texts, embeddings); err != nil {
		log.Fatalf("Failed to write output: %v", err)
	}

	// Print statistics
	if *showStats {
		fmt.Fprintf(os.Stderr, "\nStatistics:\n")
		fmt.Fprintf(os.Stderr, "  Texts processed: %d\n", len(texts))
		fmt.Fprintf(os.Stderr, "  Total time: %v\n", embedDuration)
		fmt.Fprintf(os.Stderr, "  Average time: %v per text\n", embedDuration/time.Duration(len(texts)))
		fmt.Fprintf(os.Stderr, "  Throughput: %.2f texts/sec\n", float64(len(texts))/embedDuration.Seconds())
	}
}

func writeOutput(w io.Writer, format string, texts []string, embeddings [][]float32) error {
	switch format {
	case "json":
		return writeJSON(w, texts, embeddings)
	case "csv":
		return writeCSV(w, texts, embeddings, ',')
	case "tsv":
		return writeCSV(w, texts, embeddings, '\t')
	default:
		return fmt.Errorf("unknown format: %s", format)
	}
}

func writeJSON(w io.Writer, texts []string, embeddings [][]float32) error {
	type Result struct {
		Text      string    `json:"text"`
		Embedding []float32 `json:"embedding"`
	}

	results := make([]Result, len(texts))
	for i := range texts {
		results[i] = Result{
			Text:      texts[i],
			Embedding: embeddings[i],
		}
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}

func writeCSV(w io.Writer, texts []string, embeddings [][]float32, delimiter rune) error {
	writer := csv.NewWriter(w)
	writer.Comma = delimiter

	// Write header
	if len(embeddings) > 0 && len(embeddings[0]) > 0 {
		header := []string{"text"}
		for i := 0; i < len(embeddings[0]); i++ {
			header = append(header, fmt.Sprintf("dim_%d", i))
		}
		if err := writer.Write(header); err != nil {
			return err
		}
	}

	// Write rows
	for i, text := range texts {
		row := []string{text}
		for _, val := range embeddings[i] {
			row = append(row, strconv.FormatFloat(float64(val), 'f', -1, 32))
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	writer.Flush()
	return writer.Error()
}
