package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lth/pure-go-llamas/model"
	"github.com/lth/pure-go-llamas/pkg/ggufembed"
	"github.com/lth/pure-go-llamas/search"
	annoyindex "github.com/lth/pure-go-llamas/search/annoy"
	bruteindex "github.com/lth/pure-go-llamas/search/brute"
)

const batchSize = 128

type iconRecord struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

type iconFile struct {
	Icons []iconRecord `json:"icons"`
}

func main() {
	iconsPath := flag.String("icons", "../one/go/hugeicons/icons.json", "Path to the icons dataset")
	queryText := flag.String("query", "sparkle icon", "Query to run after building the indices")
	dim := flag.Int("dim", 512, "Embedding dimension")
	trees := flag.Int("trees", 64, "Number of Annoy trees")
	leafSize := flag.Int("leaf", 32, "Maximum Annoy leaf size")
	quantFlag := flag.String("quant", "int8", "Brute quantization mode (int8|int16|fp32)")
	flag.Parse()

	icons, err := loadIcons(*iconsPath)
	if err != nil {
		log.Fatalf("load icons: %v", err)
	}

	quantMode, err := parseQuantization(*quantFlag)
	if err != nil {
		log.Fatalf("invalid quantization: %v", err)
	}

	baseName := strings.TrimSuffix(filepath.Base(*iconsPath), filepath.Ext(*iconsPath))
	annoyPath := fmt.Sprintf("%s.annoy.%d.%d.%d.idx", baseName, *dim, *trees, *leafSize)
	brutePath := fmt.Sprintf("%s.brute.%d.%s.idx", baseName, *dim, quantizationLabel(quantMode))

	needAnnoy := !fileExists(annoyPath)
	needBrute := !fileExists(brutePath)

	fmt.Printf("Dataset: %s (%d icons)\n", filepath.Base(*iconsPath), len(icons))
	fmt.Printf("Annoy index: %s (%s)\n", annoyPath, humanCacheState(needAnnoy))
	fmt.Printf("Brute index (%s): %s (%s)\n", quantizationLabel(quantMode), brutePath, humanCacheState(needBrute))

	rt, err := model.Open()
	if err != nil {
		log.Fatalf("open runtime: %v", err)
	}
	defer rt.Close()

	ctx := context.Background()

	var (
		annIdx   *annoyindex.Index
		bruteIdx *bruteindex.Index
	)

	if needAnnoy || needBrute {
		fmt.Printf("Embedding %d icons (dim=%d)…\n", len(icons), *dim)

		var annBuilder *annoyindex.Builder
		if needAnnoy {
			annBuilder = annoyindex.NewBuilder(
				annoyindex.WithDimension(*dim),
				annoyindex.WithNumTrees(*trees),
				annoyindex.WithMaxLeafSize(*leafSize),
				annoyindex.WithProgress(progressPrinter("annoy-build")),
			)
		}
		var bruteBuilder *bruteindex.Builder
		if needBrute {
			bruteBuilder = bruteindex.NewBuilder(
				bruteindex.WithDimension(*dim),
				bruteindex.WithQuantization(quantMode),
			)
		}

		embedStart := time.Now()
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
					Dim:     *dim,
				}
			}
			vectors, err := rt.EmbedInputs(ctx, inputs)
			if err != nil {
				log.Fatalf("embed inputs: %v", err)
			}
			for i, vec := range vectors {
				id := int32(start + i)
				if needAnnoy {
					if err := annBuilder.AddVector(id, vec); err != nil {
						log.Fatalf("annoy add vector: %v", err)
					}
				}
				if needBrute {
					if err := bruteBuilder.AddVector(id, vec); err != nil {
						log.Fatalf("brute add vector: %v", err)
					}
				}
			}
			progress := float64(end) / float64(len(icons)) * 100
			fmt.Printf("\r[embed] %d/%d (%.1f%%)", end, len(icons), progress)
		}
		fmt.Println()
		embedDur := time.Since(embedStart)
		fmt.Printf("Embedding complete in %s\n", formatDuration(embedDur))

		if needAnnoy {
			fmt.Printf("Building Annoy index (trees=%d, leaf=%d)…\n", *trees, *leafSize)
			buildStart := time.Now()
			annIface, err := annBuilder.Build(ctx)
			if err != nil {
				log.Fatalf("annoy build: %v", err)
			}
			annIdx = annIface.(*annoyindex.Index)
			buildDur := time.Since(buildStart)

			serialStart := time.Now()
			data, err := annIdx.Bytes()
			if err != nil {
				log.Fatalf("annoy serialize: %v", err)
			}
			if err := os.WriteFile(annoyPath, data, 0o644); err != nil {
				log.Fatalf("write annoy index: %v", err)
			}
			writeDur := time.Since(serialStart)
			fmt.Printf("Saved Annoy index to %s (%d bytes, build=%s write=%s)\n",
				annoyPath, len(data), formatDuration(buildDur), formatDuration(writeDur))
		}

		if needBrute {
			fmt.Printf("Building brute-force index (quant=%s)…\n", quantizationLabel(quantMode))
			buildStart := time.Now()
			bruteIface, err := bruteBuilder.Build(ctx)
			if err != nil {
				log.Fatalf("brute build: %v", err)
			}
			bruteIdx = bruteIface.(*bruteindex.Index)
			buildDur := time.Since(buildStart)

			var ser bruteindex.Serializer
			serialStart := time.Now()
			data, err := ser.Serialize(bruteIdx)
			if err != nil {
				log.Fatalf("brute serialize: %v", err)
			}
			if err := os.WriteFile(brutePath, data, 0o644); err != nil {
				log.Fatalf("write brute index: %v", err)
			}
			writeDur := time.Since(serialStart)
			fmt.Printf("Saved brute index to %s (%d bytes, build=%s write=%s)\n",
				brutePath, len(data), formatDuration(buildDur), formatDuration(writeDur))
		}
	}

	if annIdx == nil {
		info, err := os.Stat(annoyPath)
		if err != nil {
			log.Fatalf("stat annoy index: %v", err)
		}
		loadStart := time.Now()
		loaded, err := annoyindex.LoadFile(annoyPath)
		if err != nil {
			log.Fatalf("load annoy index: %v", err)
		}
		loadDur := time.Since(loadStart)
		fmt.Printf("Loaded Annoy index from %s in %s (%d bytes)\n",
			annoyPath, formatDuration(loadDur), info.Size())
		annIdx = loaded
	}

	if bruteIdx == nil {
		info, err := os.Stat(brutePath)
		if err != nil {
			log.Fatalf("stat brute index: %v", err)
		}
		loadStart := time.Now()
		data, err := os.ReadFile(brutePath)
		if err != nil {
			log.Fatalf("read brute index: %v", err)
		}
		var ser bruteindex.Serializer
		loadedIface, err := ser.Deserialize(data)
		if err != nil {
			log.Fatalf("deserialize brute index: %v", err)
		}
		loadDur := time.Since(loadStart)
		fmt.Printf("Loaded brute index from %s in %s (%d bytes)\n",
			brutePath, formatDuration(loadDur), info.Size())
		bruteIdx = loadedIface.(*bruteindex.Index)
	}

	queryEmbedStart := time.Now()
	queryVec, err := rt.EmbedSingleInput(ctx, ggufembed.EmbedInput{
		Task:    ggufembed.TaskSearchQuery,
		Content: *queryText,
		Dim:     *dim,
	})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	queryEmbedDur := time.Since(queryEmbedStart)

	fmt.Printf("\nQuery: %q\n", *queryText)
	fmt.Printf("Query embed time: %s\n\n", formatDuration(queryEmbedDur))
	runSearch("Annoy", annIdx, queryVec, icons)
	runSearch("Brute", bruteIdx, queryVec, icons)
}

func runSearch(label string, idx search.Index, query []float32, icons []iconRecord) {
	searchStart := time.Now()
	results, err := idx.SearchVector(query, 10)
	if err != nil {
		log.Fatalf("%s search: %v", label, err)
	}
	searchDur := time.Since(searchStart)

	fmt.Printf("%s results (search=%s):\n", label, formatDuration(searchDur))
	for i, res := range results {
		id := int(res.ID)
		if id < 0 || id >= len(icons) {
			continue
		}
		icon := icons[id]
		fmt.Printf("  %2d. (%.4f) %-20s\n", i+1, res.Distance, icon.ID)
		if icon.Title != "" {
			fmt.Printf("      %s\n", icon.Title)
		}
	}
	fmt.Println()
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

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func humanCacheState(build bool) string {
	if build {
		return "build pending"
	}
	return "reuse"
}

func quantizationBits(mode bruteindex.QuantizationMode) int {
	switch mode {
	case bruteindex.QuantizeInt8:
		return 8
	case bruteindex.QuantizeInt16:
		return 16
	case bruteindex.QuantizeFloat32:
		return 32
	default:
		return 0
	}
}

func quantizationLabel(mode bruteindex.QuantizationMode) string {
	switch mode {
	case bruteindex.QuantizeInt8:
		return "int8"
	case bruteindex.QuantizeInt16:
		return "int16"
	case bruteindex.QuantizeFloat32:
		return "fp32"
	default:
		return "unknown"
	}
}

func progressPrinter(defaultLabel string) annoyindex.ProgressFunc {
	return func(stage string, current, total int) {
		if total == 0 {
			return
		}
		label := defaultLabel
		if stage != "" {
			label = stage
		}
		percent := float64(current) / float64(total) * 100
		fmt.Printf("\r[%s] %d/%d (%.1f%%)", label, current, total, percent)
		if current == total {
			fmt.Println()
		}
	}
}

func formatDuration(d time.Duration) string {
	switch {
	case d >= time.Second:
		return fmt.Sprintf("%.2fs", d.Seconds())
	case d >= time.Millisecond:
		return fmt.Sprintf("%.1fms", float64(d)/float64(time.Millisecond))
	case d >= time.Microsecond:
		return fmt.Sprintf("%.1fµs", float64(d)/float64(time.Microsecond))
	default:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	}
}

func parseQuantization(value string) (bruteindex.QuantizationMode, error) {
	switch strings.ToLower(value) {
	case "int8", "q8":
		return bruteindex.QuantizeInt8, nil
	case "int16", "q16":
		return bruteindex.QuantizeInt16, nil
	case "fp32", "float32", "f32":
		return bruteindex.QuantizeFloat32, nil
	default:
		return 0, fmt.Errorf("unsupported quantization %q", value)
	}
}
