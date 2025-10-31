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

	"github.com/headlands-org/go-semantica/model"
	"github.com/headlands-org/go-semantica/pkg/ggufembed"
	"github.com/headlands-org/go-semantica/search"
	annoyindex "github.com/headlands-org/go-semantica/search/annoy"
)

const defaultIconsPath = "../one/go/hugeicons/icons.json"

type iconInfo struct {
	Slug        string
	Title       string
	Description string
}

type embeddingRecord struct {
	ID     int32     `json:"id"`
	Vector []float32 `json:"vector"`
}

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
	embeddingsPath := fs.String("embeddings", "", "Optional path to export embeddings JSONL")
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

	rt, err := model.Open()
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()
	progress := newProgressPrinter()
	builder := annoyindex.NewBuilder(
		annoyindex.WithNumTrees(*numTrees),
		annoyindex.WithMaxLeafSize(*maxLeaf),
		annoyindex.WithDimension(*targetDim),
		annoyindex.WithProgress(progress),
	)

	var (
		embedFile *os.File
		embedMu   sync.Mutex
	)
	if *embeddingsPath != "" {
		f, err := os.Create(*embeddingsPath)
		if err != nil {
			log.Fatalf("create embeddings file: %v", err)
		}
		embedFile = f
		defer func() {
			embedMu.Lock()
			if err := embedFile.Close(); err != nil {
				log.Printf("close embeddings file: %v", err)
			}
			embedMu.Unlock()
		}()
	}

	embedStart := time.Now()
	batchSize := 128
	for startIdx := 0; startIdx < len(icons); startIdx += batchSize {
		endIdx := startIdx + batchSize
		if endIdx > len(icons) {
			endIdx = len(icons)
		}
		batch := icons[startIdx:endIdx]
		inputs := make([]ggufembed.EmbedInput, len(batch))
		for i, icon := range batch {
			inputs[i] = ggufembed.EmbedInput{
				Task:    ggufembed.TaskSearchDocument,
				Title:   icon.Title,
				Content: icon.Description,
				Dim:     *targetDim,
			}
		}
		vectors, err := rt.EmbedInputs(ctx, inputs)
		if err != nil {
			log.Fatalf("embed batch: %v", err)
		}
		for i, vec := range vectors {
			if err := builder.AddVector(int32(startIdx+i), vec); err != nil {
				log.Fatalf("AddVector failed: %v", err)
			}
			if embedFile != nil {
				record := embeddingRecord{
					ID:     int32(startIdx + i),
					Vector: vec,
				}
				data, err := json.Marshal(record)
				if err != nil {
					log.Fatalf("marshal embedding: %v", err)
				}
				embedMu.Lock()
				if _, err := embedFile.Write(data); err != nil {
					log.Fatalf("write embedding: %v", err)
				}
				if _, err := embedFile.Write([]byte("\n")); err != nil {
					log.Fatalf("write newline: %v", err)
				}
				embedMu.Unlock()
			}
		}
	}
	embedDur := time.Since(embedStart)

	buildStart := time.Now()
	idxIface, err := builder.Build(ctx)
	if err != nil {
		log.Fatalf("build index: %v", err)
	}
	annIdx, ok := idxIface.(*annoyindex.Index)
	if !ok {
		log.Fatalf("unexpected index type %T", idxIface)
	}
	buildDur := time.Since(buildStart)

	serialStart := time.Now()
	data, err := annIdx.Bytes()
	if err != nil {
		log.Fatalf("serialise index: %v", err)
	}
	if err := os.WriteFile(*outputPath, data, 0o644); err != nil {
		log.Fatalf("write index: %v", err)
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
	iconsPath := fs.String("icons", defaultIconsPath, "Path to icons.json for metadata")
	query := fs.String("query", "", "Query text")
	topK := fs.Int("top", 5, "Number of neighbours to return")
	searchK := fs.Int("searchk", 0, "search_k override (default: trees * top)")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}
	if *query == "" {
		log.Fatal("search: -query is required")
	}

	icons, err := loadIcons(*iconsPath)
	if err != nil {
		log.Fatalf("load icons: %v", err)
	}

	loadStart := time.Now()
	idx, err := annoyindex.LoadFile(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	loadDur := time.Since(loadStart)

	rt, err := model.Open()
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()
	embedStart := time.Now()
	vec, err := rt.EmbedSingleInput(ctx, ggufembed.EmbedInput{
		Task:    ggufembed.TaskSearchQuery,
		Content: *query,
		Dim:     idx.Dimension(),
	})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	embedDur := time.Since(embedStart)

	searchStart := time.Now()
	opts := []search.SearchOption{}
	if *searchK > 0 {
		opts = append(opts, search.WithSearchK(*searchK))
	}
	results, err := idx.SearchVector(vec, *topK, opts...)
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	searchDur := time.Since(searchStart)

	fmt.Printf("Loaded index in %s (%d items)\n", loadDur.Truncate(time.Millisecond), idx.Count())
	fmt.Printf("Search \"%s\" (%d results) embed=%s search=%s\n",
		*query, len(results), embedDur.Truncate(time.Millisecond), searchDur.Truncate(time.Microsecond))
	for i, res := range results {
		id := int(res.ID)
		if id < 0 || id >= len(icons) {
			continue
		}
		info := icons[id]
		fmt.Printf("%2d. %-20s dist=%.4f\n", i+1, info.Slug, res.Distance)
		if info.Title != "" {
			fmt.Printf("    Title: %s\n", info.Title)
		}
		if info.Description != "" {
			fmt.Printf("    Desc:  %s\n", truncate(info.Description, 140))
		}
	}
}

func runBrute(args []string) {
	fs := flag.NewFlagSet("brute", flag.ExitOnError)
	indexPath := fs.String("index", "icons.ann", "Path to Annoy index file")
	iconsPath := fs.String("icons", defaultIconsPath, "Path to icons.json for metadata")
	query := fs.String("query", "", "Query text")
	topK := fs.Int("top", 5, "Number of neighbours to return")
	if err := fs.Parse(args); err != nil {
		log.Fatal(err)
	}
	if *query == "" {
		log.Fatal("brute: -query is required")
	}

	icons, err := loadIcons(*iconsPath)
	if err != nil {
		log.Fatalf("load icons: %v", err)
	}

	loadStart := time.Now()
	idx, err := annoyindex.LoadFile(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	loadDur := time.Since(loadStart)

	items := collectItems(idx)
	if len(items) == 0 {
		log.Fatalf("index is empty")
	}

	rt, err := model.Open()
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()
	embedStart := time.Now()
	vec, err := rt.EmbedSingleInput(ctx, ggufembed.EmbedInput{
		Task:    ggufembed.TaskSearchQuery,
		Content: *query,
		Dim:     idx.Dimension(),
	})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	embedDur := time.Since(embedStart)

	start := time.Now()
	results := bruteForce(vec, items, idx.Metric(), -1, *topK)
	elapsed := time.Since(start)

	fmt.Printf("Brute-force search (%d items) load=%s embed=%s search=%s\n",
		len(items), loadDur.Truncate(time.Millisecond), embedDur.Truncate(time.Millisecond), elapsed.Truncate(time.Microsecond))
	for i, res := range results {
		id := int(res.ID)
		if id < 0 || id >= len(icons) {
			continue
		}
		info := icons[id]
		fmt.Printf("%2d. %-20s dist=%.4f\n", i+1, info.Slug, res.Distance)
		if info.Title != "" {
			fmt.Printf("    Title: %s\n", info.Title)
		}
		if info.Description != "" {
			fmt.Printf("    Desc:  %s\n", truncate(info.Description, 140))
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

	idx, err := annoyindex.LoadFile(*indexPath)
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
		res, err := idx.SearchVector(query.vec, searchTop, search.WithSearchK(*searchK))
		if err != nil {
			log.Fatalf("search vector: %v", err)
		}
		annoyTime += time.Since(startAnn)

		filtered := make([]search.Result, 0, len(res))
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
	id  int32
	vec []float32
}

func collectItems(idx *annoyindex.Index) []evalItem {
	items := make([]evalItem, 0, idx.Count())
	idx.ForEach(func(id int32, vec []float32) {
		copyVec := append([]float32(nil), vec...)
		items = append(items, evalItem{id: id, vec: copyVec})
	})
	sort.Slice(items, func(i, j int) bool { return items[i].id < items[j].id })
	return items
}

func bruteForce(query []float32, items []evalItem, metric annoyindex.Metric, exclude int32, topK int) []search.Result {
	q := append([]float32(nil), query...)
	if metric == annoyindex.Cosine {
		normalizeVec(q)
	}
	type scored struct {
		id   int32
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
	results := make([]search.Result, topK)
	for i := 0; i < topK; i++ {
		results[i] = search.Result{ID: scoredList[i].id, Distance: scoredList[i].dist}
	}
	return results
}

func recallAtK(truth, approx []search.Result) float64 {
	if len(truth) == 0 {
		return 0
	}
	set := make(map[int32]struct{}, len(truth))
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

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func newProgressPrinter() annoyindex.ProgressFunc {
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

func loadIcons(path string) ([]iconInfo, error) {
	type raw struct {
		Icons []struct {
			ID          string `json:"id"`
			Title       string `json:"title"`
			Description string `json:"description"`
		} `json:"icons"`
	}

	abspath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abspath)
	if err != nil {
		return nil, err
	}
	var file raw
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	icons := make([]iconInfo, len(file.Icons))
	for i, icon := range file.Icons {
		icons[i] = iconInfo{Slug: icon.ID, Title: icon.Title, Description: icon.Description}
	}
	return icons, nil
}
