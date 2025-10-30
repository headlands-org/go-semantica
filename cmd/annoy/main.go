package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/lth/pure-go-llamas/annoy"
)

const defaultIconsPath = "../one/go/hugeicons/icons.json"

func usage() {
	fmt.Fprintf(os.Stderr, `annoy example commands:

  build   Build an Annoy index from icons.json
  search  Run a semantic similarity search over an index
  eval    Evaluate Annoy vs brute-force on the index

Example:
  %[1]s build -input %s -output icons.ann
  %[1]s search -index icons.ann -query "sparkle icon" -top 5
  %[1]s eval   -index icons.ann -top 10 -searchk 200 -samples 200

`, filepath.Base(os.Args[0]), defaultIconsPath)
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "build":
		runBuild(os.Args[2:])
	case "search":
		runSearch(os.Args[2:])
	case "brute":
		runBrute(os.Args[2:])
	case "eval":
		runEval(os.Args[2:])
	default:
		usage()
	}
}

func runBuild(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	inputPath := fs.String("input", defaultIconsPath, "Path to icons.json")
	outputPath := fs.String("output", "icons.ann", "Output index file")
	embeddingsPath := fs.String("embeddings", "", "Optional embeddings output (JSONL)")
	numTrees := fs.Int("trees", 32, "Number of Annoy trees")
	maxLeaf := fs.Int("leaf", 32, "Maximum leaf size")
	targetDim := fs.Int("dim", 512, "Embedding dimension (Matryoshka tier)")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	icons, err := loadIcons(*inputPath)
	if err != nil {
		log.Fatalf("load icons: %v", err)
	}
	loadDur := time.Since(start)

	ctx := context.Background()
	progress := newProgressPrinter()
	builder := annoy.NewBuilder(
		annoy.WithNumTrees(*numTrees),
		annoy.WithMaxLeafSize(*maxLeaf),
		annoy.WithEmbeddingDim(*targetDim),
		annoy.WithDimension(*targetDim),
		annoy.WithProgress(progress),
	)

	docs := make([]annoy.Document, len(icons))
	for i, icon := range icons {
		meta, err := json.Marshal(iconMetadata{
			Title:       icon.Title,
			Description: icon.Description,
		})
		if err != nil {
			log.Fatalf("marshal metadata for %s: %v", icon.ID, err)
		}
		docs[i] = annoy.Document{ID: icon.ID, Title: icon.Title, Text: icon.Description, Metadata: meta}
	}

	embedStart := time.Now()
	if err := builder.AddBatchTexts(ctx, docs, 128); err != nil {
		log.Fatalf("embed batch: %v", err)
	}
	embedDur := time.Since(embedStart)

	buildStart := time.Now()
	if err := builder.Build(ctx); err != nil {
		log.Fatalf("build index: %v", err)
	}
	buildDur := time.Since(buildStart)

	serialStart := time.Now()
	data, err := builder.Bytes()
	if err != nil {
		log.Fatalf("serialise index: %v", err)
	}
	if err := os.WriteFile(*outputPath, data, 0o644); err != nil {
		log.Fatalf("write index: %v", err)
	}
	if *embeddingsPath != "" {
		f, err := os.Create(*embeddingsPath)
		if err != nil {
			log.Fatalf("create embeddings file: %v", err)
		}
		if err := builder.WriteEmbeddings(f); err != nil {
			f.Close()
			log.Fatalf("write embeddings: %v", err)
		}
		if err := f.Close(); err != nil {
			log.Fatalf("close embeddings: %v", err)
		}
	}
	serialDur := time.Since(serialStart)

	total := time.Since(start)

	fmt.Printf("Built index for %d icons\n", len(icons))
	fmt.Printf("  load icons:    %s\n", loadDur.Truncate(time.Millisecond))
	fmt.Printf("  embed:         %s\n", embedDur.Truncate(time.Millisecond))
	fmt.Printf("  build trees:   %s\n", buildDur.Truncate(time.Millisecond))
	fmt.Printf("  serialise:     %s (%d bytes)\n", serialDur.Truncate(time.Millisecond), len(data))
	fmt.Printf("  total:         %s\n", total.Truncate(time.Millisecond))
	fmt.Printf("Index written to %s\n", *outputPath)
	if *embeddingsPath != "" {
		fmt.Printf("Embeddings exported to %s\n", *embeddingsPath)
	}
}

func runSearch(args []string) {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	indexPath := fs.String("index", "icons.ann", "Path to Annoy index file")
	query := fs.String("query", "", "Query text")
	topK := fs.Int("top", 5, "Number of neighbours to return")
	searchK := fs.Int("searchk", 0, "search_k override (default: trees * top)")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}
	if *query == "" {
		log.Fatal("search: -query is required")
	}

	loadStart := time.Now()
	idx, err := annoy.LoadFile(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	loadDur := time.Since(loadStart)

	ctx := context.Background()
	searchStart := time.Now()
	opts := []annoy.SearchOption{annoy.WithMetadata()}
	if *searchK > 0 {
		opts = append(opts, annoy.WithSearchK(*searchK))
	}
	results, err := idx.SearchText(ctx, *query, *topK, opts...)
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	searchDur := time.Since(searchStart)

	fmt.Printf("Loaded index in %s (%d items)\n", loadDur.Truncate(time.Millisecond), len(idx.Items()))
	fmt.Printf("Search \"%s\" (%d results) in %s\n", *query, len(results), searchDur.Truncate(time.Millisecond))
	for i, res := range results {
		var meta iconMetadata
		if len(res.Metadata) > 0 {
			_ = json.Unmarshal(res.Metadata, &meta)
		}
		fmt.Printf("%2d. %-20s dist=%.4f\n", i+1, res.ID, res.Distance)
		if meta.Title != "" {
			fmt.Printf("    Title: %s\n", meta.Title)
		}
		if meta.Description != "" {
			fmt.Printf("    Desc:  %s\n", truncate(meta.Description, 140))
		}
	}
}

func runBrute(args []string) {
	fs := flag.NewFlagSet("brute", flag.ExitOnError)
	indexPath := fs.String("index", "icons.ann", "Path to Annoy index file")
	query := fs.String("query", "", "Query text")
	topK := fs.Int("top", 5, "Number of neighbours to return")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}
	if *query == "" {
		log.Fatal("brute: -query is required")
	}

	idx, err := annoy.LoadFile(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}

	items := collectItems(idx)
	if len(items) == 0 {
		log.Fatalf("index is empty")
	}

	ctx := context.Background()
	vec, err := idx.EmbedQuery(ctx, *query)
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}

	start := time.Now()
	results := bruteForce(vec, items, idx.Metric(), "", *topK)
	elapsed := time.Since(start)

	fmt.Printf("Brute-force search (%d items) took %s\n", len(items), elapsed.Truncate(time.Microsecond))
	for i, res := range results {
		meta, _ := idx.Metadata(res.ID)
		var metaStruct iconMetadata
		if len(meta) > 0 {
			_ = json.Unmarshal(meta, &metaStruct)
		}
		fmt.Printf("%2d. %-20s dist=%.4f\n", i+1, res.ID, res.Distance)
		if metaStruct.Title != "" {
			fmt.Printf("    Title: %s\n", metaStruct.Title)
		}
		if metaStruct.Description != "" {
			fmt.Printf("    Desc:  %s\n", truncate(metaStruct.Description, 140))
		}
	}
}

func runEval(args []string) {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	indexPath := fs.String("index", "icons.ann", "Path to Annoy index file")
	topK := fs.Int("top", 10, "Nearest neighbours to evaluate")
	searchK := fs.Int("searchk", 0, "search_k override (default: trees * top)")
	samples := fs.Int("samples", 200, "Number of items to sample (<=0 = all)")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}

	idx, err := annoy.LoadFile(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}

	items := collectItems(idx)
	if len(items) == 0 {
		log.Fatalf("index is empty")
	}

	n := *samples
	if n <= 0 || n > len(items) {
		n = len(items)
	}

	searchTop := *topK + 1

	fmt.Printf("Evaluating %d queries (top=%d searchK=%d)\n", n, *topK, *searchK)

	var annoyTime, bruteTime time.Duration
	var recallSum float64

	for i := 0; i < n; i++ {
		query := items[i]

		startBF := time.Now()
		brute := bruteForce(query.vec, items, idx.Metric(), query.id, *topK)
		bruteTime += time.Since(startBF)

		startAnn := time.Now()
		res, err := idx.SearchVector(query.vec, searchTop, annoy.WithSearchK(*searchK))
		if err != nil {
			log.Fatalf("search vector: %v", err)
		}
		annoyTime += time.Since(startAnn)

		filtered := make([]annoy.Result, 0, len(res))
		for _, cand := range res {
			if cand.ID == query.id {
				continue
			}
			filtered = append(filtered, cand)
			if len(filtered) == *topK {
				break
			}
		}

		recall := recallAtK(brute, filtered)
		recallSum += recall
	}

	avgRecall := recallSum / float64(n)
	fmt.Printf("  brute-force avg time: %s\n", (bruteTime / time.Duration(n)).Truncate(time.Microsecond))
	fmt.Printf("  annoy      avg time: %s\n", (annoyTime / time.Duration(n)).Truncate(time.Microsecond))
	fmt.Printf("  recall@%d:          %.2f%%\n", *topK, avgRecall*100)
}

type evalItem struct {
	id  string
	vec []float32
}

func collectItems(idx *annoy.Index) []evalItem {
	items := make([]evalItem, 0, len(idx.Items()))
	idx.ForEachVector(func(id string, vec []float32) {
		copyVec := append([]float32(nil), vec...)
		items = append(items, evalItem{id: id, vec: copyVec})
	})
	return items
}

func bruteForce(query []float32, items []evalItem, metric annoy.Metric, exclude string, topK int) []annoy.Result {
	q := append([]float32(nil), query...)
	if metric == annoy.Cosine {
		normalizeVec(q)
	}
	type scored struct {
		id   string
		dist float32
	}
	scoredList := make([]scored, 0, len(items))
	for _, item := range items {
		if item.id == exclude {
			continue
		}
		dist := metric.Distance(q, item.vec)
		scoredList = append(scoredList, scored{id: item.id, dist: dist})
	}
	sort.Slice(scoredList, func(i, j int) bool {
		if scoredList[i].dist == scoredList[j].dist {
			return scoredList[i].id < scoredList[j].id
		}
		return scoredList[i].dist < scoredList[j].dist
	})
	if topK > len(scoredList) {
		topK = len(scoredList)
	}
	results := make([]annoy.Result, topK)
	for i := 0; i < topK; i++ {
		results[i] = annoy.Result{ID: scoredList[i].id, Distance: scoredList[i].dist}
	}
	return results
}

func recallAtK(truth, approx []annoy.Result) float64 {
	if len(truth) == 0 {
		return 0
	}
	set := make(map[string]struct{}, len(truth))
	for _, res := range truth {
		set[res.ID] = struct{}{}
	}
	hits := 0
	for _, res := range approx {
		if _, ok := set[res.ID]; ok {
			hits++
		}
	}
	denom := len(truth)
	if len(approx) < denom {
		denom = len(approx)
	}
	return float64(hits) / float64(denom)
}

func normalizeVec(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

type icon struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

type iconMetadata struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}

type iconFile struct {
	Icons []icon `json:"icons"`
}

func loadIcons(path string) ([]icon, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file iconFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	return file.Icons, nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func newProgressPrinter() annoy.ProgressFunc {
	var mu sync.Mutex
	var lastStage string
	return func(stage string, current, total int) {
		if total == 0 {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if stage != lastStage {
			if lastStage != "" {
				fmt.Println()
			}
			lastStage = stage
		}
		pct := float64(current) / float64(total) * 100
		fmt.Printf("\r[%s] %d/%d (%.1f%%)", stage, current, total, pct)
		if current == total {
			fmt.Println()
		}
	}
}
