package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/lth/pure-go-llamas/model"
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
	"github.com/lth/pure-go-llamas/search"
	brute "github.com/lth/pure-go-llamas/search/brute"
)

const (
	defaultIconsPath = "../one/go/hugeicons/icons.json"
	topK             = 10
)

type iconRecord struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

type iconFile struct {
	Icons []iconRecord `json:"icons"`
}

type metric struct {
	dimension   int
	quantLabel  string
	indexBytes  int64
	avgSearchUs float64
	recallPct   float64
}

type target struct {
	dim   int
	mode  brute.QuantizationMode
	label string
}

func main() {
	iconsPath := flag.String("icons", defaultIconsPath, "Path to icons JSON dataset")
	samples := flag.Int("samples", 500, "Number of random queries to evaluate")
	seed := flag.Int64("seed", 42, "Random seed for sampling queries")
	format := flag.String("format", "markdown", "Output format: markdown|csv")
	flag.Parse()

	icons, err := loadIcons(*iconsPath)
	if err != nil {
		log.Fatalf("load icons: %v", err)
	}

	vectors768, err := embedIcons(icons)
	if err != nil {
		log.Fatalf("embed icons: %v", err)
	}

	dims := []int{768, 512, 256, 128}
	vectorsByDim := make(map[int][][]float32, len(dims))
	for _, dim := range dims {
		truncated := make([][]float32, len(vectors768))
		for i, base := range vectors768 {
			vec := make([]float32, dim)
			copy(vec, base[:dim])
			normalizeVec(vec)
			truncated[i] = vec
		}
		vectorsByDim[dim] = truncated
	}

	targets := []target{
		{768, brute.QuantizeFloat32, "fp32"},
		{768, brute.QuantizeInt16, "int16"},
		{768, brute.QuantizeInt8, "int8"},
		{512, brute.QuantizeFloat32, "fp32"},
		{512, brute.QuantizeInt16, "int16"},
		{512, brute.QuantizeInt8, "int8"},
		{256, brute.QuantizeFloat32, "fp32"},
		{256, brute.QuantizeInt16, "int16"},
		{256, brute.QuantizeInt8, "int8"},
		{128, brute.QuantizeFloat32, "fp32"},
		{128, brute.QuantizeInt16, "int16"},
		{128, brute.QuantizeInt8, "int8"},
	}

	baselineIdx, baselineSize, err := buildIndex(vectorsByDim[768], 768, brute.QuantizeFloat32)
	if err != nil {
		log.Fatalf("build baseline index: %v", err)
	}

	ids, truth, queryVectors := prepareQueries(baselineIdx, vectors768, *samples, *seed)

	metrics := make([]metric, 0, len(targets))
	metrics = append(metrics, metric{
		dimension:   768,
		quantLabel:  "fp32",
		indexBytes:  baselineSize,
		avgSearchUs: averageSearch(baselineIdx, ids, queryVectors[768], truth),
		recallPct:   100.0,
	})

	for _, tgt := range targets {
		if tgt.dim == 768 && tgt.mode == brute.QuantizeFloat32 {
			continue
		}
		idx, size, err := buildIndex(vectorsByDim[tgt.dim], tgt.dim, tgt.mode)
		if err != nil {
			log.Fatalf("build index %d %s: %v", tgt.dim, tgt.label, err)
		}
		avgUs, recall := evaluateIndex(idx, ids, queryVectors[tgt.dim], truth)
		metrics = append(metrics, metric{
			dimension:   tgt.dim,
			quantLabel:  tgt.label,
			indexBytes:  size,
			avgSearchUs: avgUs,
			recallPct:   recall,
		})
	}

	switch *format {
	case "markdown":
		printMarkdown(metrics)
	case "csv":
		printCSV(metrics)
	default:
		log.Fatalf("unknown format: %s", *format)
	}
}

func loadIcons(path string) ([]iconRecord, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	var file iconFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	return file.Icons, nil
}

func embedIcons(icons []iconRecord) ([][]float32, error) {
	rt, err := model.Open()
	if err != nil {
		return nil, err
	}
	defer rt.Close()

	ctx := context.Background()
	const dim = 768
	batchSize := 128
	vectors := make([][]float32, len(icons))

	for start := 0; start < len(icons); start += batchSize {
		end := start + batchSize
		if end > len(icons) {
			end = len(icons)
		}
		batch := icons[start:end]
		inputs := make([]ggufembed.EmbedInput, len(batch))
		for i, icon := range batch {
			inputs[i] = ggufembed.EmbedInput{
				Task:    ggufembed.TaskSearchDocument,
				Title:   icon.Title,
				Content: icon.Description,
				Dim:     dim,
			}
		}
		out, err := rt.EmbedInputs(ctx, inputs)
		if err != nil {
			return nil, err
		}
		for i, vec := range out {
			copyVec := make([]float32, len(vec))
			copy(copyVec, vec)
			normalizeVec(copyVec)
			vectors[start+i] = copyVec
		}
	}
	return vectors, nil
}

func buildIndex(vectors [][]float32, dim int, mode brute.QuantizationMode) (*brute.Index, int64, error) {
	builder := brute.NewBuilder(
		brute.WithDimension(dim),
		brute.WithQuantization(mode),
	)
	for i, vec := range vectors {
		if err := builder.AddVector(int32(i), vec); err != nil {
			return nil, 0, err
		}
	}
	idxIface, err := builder.Build(context.Background())
	if err != nil {
		return nil, 0, err
	}
	idx := idxIface.(*brute.Index)
	var ser brute.Serializer
	data, err := ser.Serialize(idx)
	if err != nil {
		return nil, 0, err
	}
	return idx, int64(len(data)), nil
}

func prepareQueries(baseline *brute.Index, baseVectors [][]float32, sampleCount int, seed int64) ([]int32, [][]int32, map[int][][]float32) {
	total := len(baseVectors)
	if sampleCount > total {
		sampleCount = total
	}
	rng := rand.New(rand.NewSource(seed))
	perm := rng.Perm(total)
	ids := make([]int32, sampleCount)
	queriesByDim := make(map[int][][]float32)
	dims := []int{768, 512, 256, 128}
	for _, dim := range dims {
		queriesByDim[dim] = make([][]float32, sampleCount)
	}
	truth := make([][]int32, sampleCount)

	for i := 0; i < sampleCount; i++ {
		id := perm[i]
		ids[i] = int32(id)
		baseVec := make([]float32, len(baseVectors[id]))
		copy(baseVec, baseVectors[id])
		normalizeVec(baseVec)

		baseResults, err := baseline.SearchNormalized(baseVec, topK+1)
		if err != nil {
			log.Fatalf("baseline search: %v", err)
		}
		truth[i] = filterResults(baseResults, ids[i])

		for _, dim := range dims {
			truncated := make([]float32, dim)
			copy(truncated, baseVec[:dim])
			normalizeVec(truncated)
			queriesByDim[dim][i] = truncated
		}
	}
	return ids, truth, queriesByDim
}

func evaluateIndex(idx *brute.Index, ids []int32, queries [][]float32, truth [][]int32) (avgUs float64, recall float64) {
	var total time.Duration
	var recallSum float64
	for i, id := range ids {
		start := time.Now()
		res, err := idx.SearchNormalized(queries[i], topK+1)
		if err != nil {
			log.Fatalf("search: %v", err)
		}
		total += time.Since(start)
		filtered := filterResults(res, id)
		recallSum += recallAtK(truth[i], filtered)
	}
	avgUs = float64(total.Microseconds()) / float64(len(ids))
	recall = (recallSum / float64(len(ids))) * 100
	return
}

func averageSearch(idx *brute.Index, ids []int32, queries [][]float32, truth [][]int32) float64 {
	avgUs, _ := evaluateIndex(idx, ids, queries, truth)
	return avgUs
}

func filterResults(results []search.Result, self int32) []int32 {
	filtered := make([]int32, 0, topK)
	for _, r := range results {
		if r.ID == self {
			continue
		}
		filtered = append(filtered, r.ID)
		if len(filtered) == topK {
			break
		}
	}
	return filtered
}

func recallAtK(truth, candidates []int32) float64 {
	set := make(map[int32]struct{}, len(truth))
	for _, id := range truth {
		set[id] = struct{}{}
	}
	hits := 0
	for _, id := range candidates {
		if _, ok := set[id]; ok {
			hits++
		}
	}
	denom := len(truth)
	if len(candidates) < denom {
		denom = len(candidates)
	}
	if denom == 0 {
		return 0
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

func printMarkdown(metrics []metric) {
	fmt.Println("| Dim | Quant | Index Size | Avg Search (ms) | Recall vs 768 fp32 | Recall Loss |")
	fmt.Println("|-----|-------|------------|-----------------|--------------------|-------------|")
	for _, m := range metrics {
		loss := 100.0 - m.recallPct
		fmt.Printf("| %3d | %-5s | %6.2f MB | %7.2f | %6.2f %% | %6.2f %% |\n",
			m.dimension,
			m.quantLabel,
			float64(m.indexBytes)/(1024.0*1024.0),
			m.avgSearchUs/1000.0,
			m.recallPct,
			loss,
		)
	}
}

func printCSV(metrics []metric) {
	fmt.Println("dimension,quantization,index_mb,avg_search_ms,recall_pct,recall_loss")
	for _, m := range metrics {
		loss := 100.0 - m.recallPct
		fmt.Printf("%d,%s,%.3f,%.4f,%.3f,%.3f\n",
			m.dimension,
			m.quantLabel,
			float64(m.indexBytes)/(1024.0*1024.0),
			m.avgSearchUs/1000.0,
			m.recallPct,
			loss,
		)
	}
}
