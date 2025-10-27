// Command gemma-benchmark benchmarks embedding generation performance
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lth/pure-go-llamas/pkg/ggufembed"
)

var (
	// Required flags
	modelPath = flag.String("model", "", "Path to GGUF model file (required)")
	duration  = flag.Int("duration", 0, "Benchmark duration in seconds (required for batch/isolated modes)")
	mode      = flag.String("mode", "", "Benchmark mode: batch, isolated, or comprehensive (required)")

	// Optional flags
	workers      = flag.Int("workers", runtime.NumCPU()*2, "Number of worker goroutines")
	batchSize    = flag.Int("batch-size", runtime.NumCPU()*4, "Batch size for batch mode")
	cpuProfile   = flag.String("cpuprofile", "", "Write CPU profile to file")
	blockProfile = flag.String("blockprofile", "", "Write blocking profile to file")
	mutexProfile = flag.String("mutexprofile", "", "Write mutex contention profile to file")
)

// Test documents for the 5-scenario benchmark matrix

// shortDoc is a 9-word document for short document tests
const shortDoc = "The quick brown fox jumps over the lazy dog"

// longDoc is a 49-word document for long document tests
const longDoc = "Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."

// shortDocs is a set of 5 varied short sentences (6-9 words each) for batch tests
var shortDocs = []string{
	"The quick brown fox jumps over the lazy dog",
	"Artificial intelligence is transforming modern technology",
	"Machine learning enables computers to learn from data",
	"Neural networks process information efficiently",
	"Deep learning powers many AI applications",
}

// testTexts is kept for batch/isolated modes (legacy)
var testTexts = []string{
	"Machine learning enables computers to learn from data without explicit programming",
	"Neural networks use layers of interconnected nodes to process complex patterns",
	"Distributed systems coordinate multiple computers to solve problems at scale",
	"Encryption algorithms protect sensitive data through mathematical transformations",
	"Quantum computing leverages quantum mechanics to solve certain problems exponentially faster",
	"Graph databases optimize storage and retrieval of highly connected data",
	"Microservices architecture decomposes applications into independent deployable services",
	"Container orchestration automates deployment scaling and management of containerized applications",
	"Natural language processing enables computers to understand and generate human language",
	"Computer vision algorithms extract meaningful information from digital images and videos",
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Benchmark embedding generation performance.\n\n")
		fmt.Fprintf(os.Stderr, "Required flags:\n")
		fmt.Fprintf(os.Stderr, "  -model string\n")
		fmt.Fprintf(os.Stderr, "        Path to GGUF model file\n")
		fmt.Fprintf(os.Stderr, "  -duration int\n")
		fmt.Fprintf(os.Stderr, "        Benchmark duration in seconds (not required for comprehensive mode)\n")
		fmt.Fprintf(os.Stderr, "  -mode string\n")
		fmt.Fprintf(os.Stderr, "        Benchmark mode: 'batch', 'isolated', or 'comprehensive'\n")
		fmt.Fprintf(os.Stderr, "\nOptional flags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample commands:\n")
		fmt.Fprintf(os.Stderr, "  # Batch mode with 10 second run\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode batch -duration 10\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Isolated mode with custom workers and duration\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode isolated -duration 30 -workers 8\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Comprehensive benchmark for README documentation\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Batch mode with CPU profiling\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode batch -duration 10 -cpuprofile cpu.prof\n", os.Args[0])
	}

	flag.Parse()

	// Validate required flags
	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -model is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if *mode != "batch" && *mode != "isolated" && *mode != "comprehensive" {
		fmt.Fprintf(os.Stderr, "Error: -mode must be 'batch', 'isolated', or 'comprehensive'\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Duration is required for batch and isolated modes, but not comprehensive
	if (*mode == "batch" || *mode == "isolated") && *duration <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -duration must be greater than 0 for batch/isolated modes\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Setup profiling
	cleanupProfiling := setupProfiling()
	defer cleanupProfiling()

	// Handle comprehensive mode separately
	if *mode == "comprehensive" {
		runComprehensiveMode()
		return
	}

	// Load model for batch/isolated modes
	log.Printf("Loading model from %s...", *modelPath)
	startLoad := time.Now()
	rt, err := ggufembed.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()
	log.Printf("Model loaded in %v", time.Since(startLoad))
	log.Printf("Embedding dimension: %d", rt.EmbedDim())
	log.Printf("Max sequence length: %d", rt.MaxSeqLen())

	// Run benchmark
	var embeddings int
	var benchDuration time.Duration
	var workerCounts []int

	log.Printf("Running %s mode benchmark for %d seconds...", *mode, *duration)
	log.Printf("Workers: %d, Batch size: %d", *workers, *batchSize)

	switch *mode {
	case "batch":
		embeddings, benchDuration = runBatchMode(rt)
	case "isolated":
		embeddings, benchDuration, workerCounts = runIsolatedMode(rt)
	}

	// Collect memory statistics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Report results to stderr
	fmt.Fprintf(os.Stderr, "\n=== Benchmark Results ===\n")
	fmt.Fprintf(os.Stderr, "Mode: %s\n", *mode)
	fmt.Fprintf(os.Stderr, "Duration: %v\n", benchDuration)
	fmt.Fprintf(os.Stderr, "Total embeddings: %d\n", embeddings)
	fmt.Fprintf(os.Stderr, "Throughput: %.2f embeddings/sec\n", float64(embeddings)/benchDuration.Seconds())
	fmt.Fprintf(os.Stderr, "Average latency: %v per embedding\n", benchDuration/time.Duration(embeddings))

	// Per-worker statistics for isolated mode
	if *mode == "isolated" && len(workerCounts) > 0 {
		fmt.Fprintf(os.Stderr, "\n=== Per-Worker Statistics ===\n")
		minCount := workerCounts[0]
		maxCount := workerCounts[0]
		totalCount := 0
		for _, count := range workerCounts {
			if count < minCount {
				minCount = count
			}
			if count > maxCount {
				maxCount = count
			}
			totalCount += count
		}
		avgCount := float64(totalCount) / float64(len(workerCounts))
		fmt.Fprintf(os.Stderr, "Workers: %d\n", len(workerCounts))
		fmt.Fprintf(os.Stderr, "Min embeddings per worker: %d\n", minCount)
		fmt.Fprintf(os.Stderr, "Max embeddings per worker: %d\n", maxCount)
		fmt.Fprintf(os.Stderr, "Avg embeddings per worker: %.2f\n", avgCount)
	}

	// Memory statistics
	fmt.Fprintf(os.Stderr, "\n=== Memory Statistics ===\n")
	fmt.Fprintf(os.Stderr, "HeapAlloc: %.2f MB\n", float64(m.HeapAlloc)/1024/1024)
	fmt.Fprintf(os.Stderr, "TotalAlloc: %.2f MB\n", float64(m.TotalAlloc)/1024/1024)
	fmt.Fprintf(os.Stderr, "NumGC: %d\n", m.NumGC)
}

// setupProfiling initializes profiling based on command-line flags
func setupProfiling() func() {
	var cleanups []func()

	// CPU profiling
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Printf("Warning: could not create CPU profile: %v", err)
		} else {
			log.Printf("CPU profiling enabled, writing to %s", *cpuProfile)
			if err := pprof.StartCPUProfile(f); err != nil {
				log.Printf("Warning: could not start CPU profile: %v", err)
				f.Close()
			} else {
				cleanups = append(cleanups, func() {
					pprof.StopCPUProfile()
					f.Close()
					log.Printf("CPU profile written to %s", *cpuProfile)
				})
			}
		}
	}

	// Block profiling
	if *blockProfile != "" {
		runtime.SetBlockProfileRate(1)
		log.Printf("Block profiling enabled, will write to %s", *blockProfile)
		cleanups = append(cleanups, func() {
			if err := writeProfile(*blockProfile, "block"); err != nil {
				log.Printf("Warning: could not write block profile: %v", err)
			} else {
				log.Printf("Block profile written to %s", *blockProfile)
			}
			runtime.SetBlockProfileRate(0)
		})
	}

	// Mutex profiling
	if *mutexProfile != "" {
		runtime.SetMutexProfileFraction(1)
		log.Printf("Mutex profiling enabled, will write to %s", *mutexProfile)
		cleanups = append(cleanups, func() {
			if err := writeProfile(*mutexProfile, "mutex"); err != nil {
				log.Printf("Warning: could not write mutex profile: %v", err)
			} else {
				log.Printf("Mutex profile written to %s", *mutexProfile)
			}
			runtime.SetMutexProfileFraction(0)
		})
	}

	// Return cleanup function that runs all registered cleanups
	return func() {
		for i := len(cleanups) - 1; i >= 0; i-- {
			cleanups[i]()
		}
	}
}

// writeProfile writes a named profile to a file
func writeProfile(filename, profileName string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	return pprof.Lookup(profileName).WriteTo(f, 0)
}

// runBatchMode runs the batch benchmark mode
// Returns (total embeddings generated, actual duration)
func runBatchMode(rt ggufembed.Runtime) (int, time.Duration) {
	// Set up context with deadline for precise duration control
	startTime := time.Now()
	deadline := startTime.Add(time.Duration(*duration) * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	totalEmbeddings := 0

	// Run benchmark loop until deadline is reached
	for {
		// Check if context is done before starting new iteration
		select {
		case <-ctx.Done():
			// Context deadline reached, return results
			actualDuration := time.Since(startTime)
			return totalEmbeddings, actualDuration
		default:
			// Continue with benchmark iteration
		}

		// Select batch-size random texts from the corpus
		selectedTexts := make([]string, *batchSize)
		for i := 0; i < *batchSize; i++ {
			selectedTexts[i] = testTexts[rand.Intn(len(testTexts))]
		}

		// Generate embeddings for the batch
		_, err := rt.Embed(ctx, selectedTexts)
		if err != nil {
			// Check if error is due to context cancellation
			if ctx.Err() != nil {
				// Context was cancelled/deadline reached, return results
				actualDuration := time.Since(startTime)
				return totalEmbeddings, actualDuration
			}
			// Other error, log and continue
			log.Printf("Warning: embedding failed: %v", err)
			continue
		}

		// Track total embeddings generated
		totalEmbeddings += len(selectedTexts)
	}
}

// workerResult contains results from a single worker
type workerResult struct {
	workerID int
	count    int
}

// runIsolatedMode runs the isolated benchmark mode
// Returns (total embeddings generated, actual duration, per-worker counts)
func runIsolatedMode(rt ggufembed.Runtime) (int, time.Duration, []int) {
	// Note: rt parameter is unused in isolated mode - each worker loads its own model
	_ = rt

	// Set up timing and context
	startTime := time.Now()
	deadline := startTime.Add(time.Duration(*duration) * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	// Create buffered results channel and wait group
	results := make(chan workerResult, *workers)
	var wg sync.WaitGroup

	// Spawn worker goroutines
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go worker(ctx, i, &wg, results)
	}

	// Wait for all workers to complete
	wg.Wait()
	close(results)

	// Aggregate results from all workers
	workerCounts := make([]int, *workers)
	totalEmbeddings := 0
	for result := range results {
		workerCounts[result.workerID] = result.count
		totalEmbeddings += result.count
	}

	actualDuration := time.Since(startTime)
	return totalEmbeddings, actualDuration, workerCounts
}

// worker is a goroutine that independently loads a model and generates embeddings
func worker(ctx context.Context, workerID int, wg *sync.WaitGroup, results chan<- workerResult) {
	defer wg.Done()

	// Each worker loads its own independent model instance
	rt, err := ggufembed.Open(*modelPath)
	if err != nil {
		log.Printf("Worker %d: failed to open model: %v", workerID, err)
		results <- workerResult{workerID: workerID, count: 0}
		return
	}
	defer rt.Close()

	log.Printf("Worker %d: model loaded, starting benchmark", workerID)

	// Track embeddings generated by this worker
	count := 0

	// Generate embeddings until context deadline is reached
	for {
		// Check if context is done
		select {
		case <-ctx.Done():
			// Context deadline reached, send results and return
			log.Printf("Worker %d: completed %d embeddings", workerID, count)
			results <- workerResult{workerID: workerID, count: count}
			return
		default:
			// Continue with embedding generation
		}

		// Select random text from corpus
		text := testTexts[rand.Intn(len(testTexts))]

		// Generate single embedding
		_, err := rt.EmbedSingle(ctx, text)
		if err != nil {
			// Check if error is due to context cancellation
			if ctx.Err() != nil {
				// Context was cancelled/deadline reached, send results and return
				log.Printf("Worker %d: completed %d embeddings", workerID, count)
				results <- workerResult{workerID: workerID, count: count}
				return
			}
			// Other error, log and continue
			log.Printf("Worker %d: embedding failed: %v", workerID, err)
			continue
		}

		count++
	}
}

// PlatformInfo contains platform detection information
type PlatformInfo struct {
	CPU   string
	Cores int
	OS    string
	Arch  string
}

// detectPlatform detects CPU, core count, OS, and architecture
func detectPlatform() PlatformInfo {
	info := PlatformInfo{
		Cores: runtime.NumCPU(),
		OS:    runtime.GOOS,
		Arch:  runtime.GOARCH,
	}

	// Detect CPU model
	switch runtime.GOOS {
	case "linux":
		info.CPU = detectCPULinux()
	case "darwin":
		info.CPU = detectCPUMacOS()
	default:
		info.CPU = "Unknown"
	}

	return info
}

// detectCPULinux reads CPU model from /proc/cpuinfo on Linux
func detectCPULinux() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return "Unknown"
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "model name") {
			parts := strings.Split(line, ":")
			if len(parts) >= 2 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "Unknown"
}

// detectCPUMacOS reads CPU model using sysctl on macOS
func detectCPUMacOS() string {
	// Try to read from sysctl
	// Note: This requires executing external command, which we avoid for simplicity
	// Instead, we'll just return a generic identifier
	if runtime.GOARCH == "arm64" {
		return "Apple Silicon"
	}
	return "Intel"
}

// LatencyStats contains latency statistics
type LatencyStats struct {
	Mean float64
	P50  float64
	P95  float64
	P99  float64
}

// ThroughputStats contains throughput statistics
type ThroughputStats struct {
	Throughput       float64
	PeakMemoryMB     float64
	Duration         float64
	TotalEmbeddings  int
}

// measureIdleMemory measures memory after model load with GC
func measureIdleMemory(rt ggufembed.Runtime) uint64 {
	// Force GC
	runtime.GC()

	// Wait for stability
	time.Sleep(100 * time.Millisecond)

	// Read memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return m.HeapAlloc
}

// warmup runs warmup embeddings to warm up caches
func warmup(rt ggufembed.Runtime, doc string) {
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingle(ctx, doc)
	}
}

// measureSingleDocLatency measures single document latency with percentiles
func measureSingleDocLatency(rt ggufembed.Runtime, doc string) LatencyStats {
	ctx := context.Background()
	const numRuns = 20

	latencies := make([]float64, numRuns)

	for i := 0; i < numRuns; i++ {
		start := time.Now()
		_, err := rt.EmbedSingle(ctx, doc)
		if err != nil {
			log.Printf("Warning: embedding failed during latency test: %v", err)
			latencies[i] = 0
			continue
		}
		latencies[i] = float64(time.Since(start).Microseconds()) / 1000.0 // Convert to ms
	}

	// Sort for percentile calculation
	sort.Float64s(latencies)

	// Calculate statistics
	mean := 0.0
	for _, l := range latencies {
		mean += l
	}
	mean /= float64(numRuns)

	// Calculate percentile indices (subtract 1 for 0-based indexing)
	numRunsFloat := float64(numRuns)
	idx50 := int(numRunsFloat * 0.50)
	if idx50 >= numRuns {
		idx50 = numRuns - 1
	}
	idx95 := int(numRunsFloat * 0.95)
	if idx95 >= numRuns {
		idx95 = numRuns - 1
	}
	idx99 := int(numRunsFloat * 0.99)
	if idx99 >= numRuns {
		idx99 = numRuns - 1
	}

	p50 := latencies[idx50]
	p95 := latencies[idx95]
	p99 := latencies[idx99]

	return LatencyStats{
		Mean: mean,
		P50:  p50,
		P95:  p95,
		P99:  p99,
	}
}

// measureThroughput runs throughput test with memory sampling
// docs can be a slice of varied documents (for short doc test) or a single document repeated (for long doc test)
func measureThroughput(rt ggufembed.Runtime, docs []string, batchSizeOverride int, durationSec int) ThroughputStats {
	ctx := context.Background()

	// Track peak memory in background goroutine
	var peakMemory uint64
	done := make(chan bool)

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				var m runtime.MemStats
				runtime.ReadMemStats(&m)
				if m.HeapAlloc > peakMemory {
					atomic.StoreUint64(&peakMemory, m.HeapAlloc)
				}
			}
		}
	}()

	// Run throughput test
	startTime := time.Now()
	deadline := startTime.Add(time.Duration(durationSec) * time.Second)
	totalEmbeddings := 0

	for time.Now().Before(deadline) {
		// Select batch-size random texts from docs
		selectedTexts := make([]string, batchSizeOverride)
		for i := 0; i < batchSizeOverride; i++ {
			selectedTexts[i] = docs[rand.Intn(len(docs))]
		}

		// Generate embeddings
		_, err := rt.Embed(ctx, selectedTexts)
		if err != nil {
			log.Printf("Warning: embedding failed during throughput test: %v", err)
			continue
		}

		totalEmbeddings += len(selectedTexts)
	}

	actualDuration := time.Since(startTime).Seconds()

	// Stop memory sampling
	close(done)
	time.Sleep(150 * time.Millisecond) // Wait for goroutine to finish

	peak := atomic.LoadUint64(&peakMemory)

	return ThroughputStats{
		Throughput:      float64(totalEmbeddings) / actualDuration,
		PeakMemoryMB:    float64(peak) / 1024 / 1024,
		Duration:        actualDuration,
		TotalEmbeddings: totalEmbeddings,
	}
}

// formatComprehensiveResults formats the 5-scenario benchmark results
func formatComprehensiveResults(platform PlatformInfo, idleMemMB float64,
	shortLatency LatencyStats, longLatency LatencyStats,
	shortThroughput ThroughputStats, longThroughput ThroughputStats) {

	fmt.Printf("\n=== Benchmark Results ===\n\n")

	fmt.Printf("Platform: %s, %d cores, %s/%s\n\n", platform.CPU, platform.Cores, platform.OS, platform.Arch)

	fmt.Printf("%-32s%-20s%-12s%s\n", "Scenario", "Metric", "Value", "Unit")
	fmt.Printf("------------------------------------------------------------------------\n")

	// Scenario 1: Idle Memory
	fmt.Printf("%-32s%-20s%-12.0f%s\n", "Idle Memory", "Heap Allocated", idleMemMB, "MB")
	fmt.Printf("\n")

	// Scenario 2: Single Short Doc (9w)
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Single Short Doc (9w)", "P50 Latency", shortLatency.P50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P95 Latency", shortLatency.P95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P99 Latency", shortLatency.P99, "ms")
	fmt.Printf("\n")

	// Scenario 3: Single Long Doc (49w)
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Single Long Doc (49w)", "P50 Latency", longLatency.P50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P95 Latency", longLatency.P95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P99 Latency", longLatency.P99, "ms")
	fmt.Printf("\n")

	// Scenario 4: Batch Short Docs (96x)
	avgLatencyShort := 1000.0 / shortThroughput.Throughput // Convert to ms/emb
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Batch Short Docs (96x)", "Throughput", shortThroughput.Throughput, "emb/sec")
	fmt.Printf("%-32s%-20s%-12.0f%s\n", "", "Peak Memory", shortThroughput.PeakMemoryMB, "MB")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "Avg Latency", avgLatencyShort, "ms/emb")
	fmt.Printf("\n")

	// Scenario 5: Batch Long Docs (96x)
	avgLatencyLong := 1000.0 / longThroughput.Throughput // Convert to ms/emb
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Batch Long Docs (96x)", "Throughput", longThroughput.Throughput, "emb/sec")
	fmt.Printf("%-32s%-20s%-12.0f%s\n", "", "Peak Memory", longThroughput.PeakMemoryMB, "MB")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "Avg Latency", avgLatencyLong, "ms/emb")
}

// runComprehensiveMode runs the 5-scenario benchmark suite
func runComprehensiveMode() {
	log.Printf("Running 5-scenario benchmark suite...")

	// 1. Platform detection
	log.Printf("[1/6] Detecting platform...")
	platform := detectPlatform()
	log.Printf("Detected: %s, %d cores, %s/%s", platform.CPU, platform.Cores, platform.OS, platform.Arch)

	// 2. Load model and measure idle memory
	log.Printf("[2/6] Loading model and measuring idle memory...")
	rt, err := ggufembed.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	idleMemBytes := measureIdleMemory(rt)
	idleMemMB := float64(idleMemBytes) / 1024 / 1024
	log.Printf("Scenario 1 - Idle memory: %.0f MB", idleMemMB)

	// 3. Single Short Doc (9w) - Warmup + Measure
	log.Printf("[3/6] Scenario 2 - Single short doc (9w): warmup + 20 runs...")
	warmup(rt, shortDoc)
	shortLatency := measureSingleDocLatency(rt, shortDoc)
	log.Printf("Scenario 2 - Short doc P50: %.1f ms, P95: %.1f ms, P99: %.1f ms",
		shortLatency.P50, shortLatency.P95, shortLatency.P99)

	// 4. Single Long Doc (49w) - Warmup + Measure
	log.Printf("[4/6] Scenario 3 - Single long doc (49w): warmup + 20 runs...")
	warmup(rt, longDoc)
	longLatency := measureSingleDocLatency(rt, longDoc)
	log.Printf("Scenario 3 - Long doc P50: %.1f ms, P95: %.1f ms, P99: %.1f ms",
		longLatency.P50, longLatency.P95, longLatency.P99)

	// 5. Batch Short Docs (96x) - 20 seconds
	log.Printf("[5/6] Scenario 4 - Batch short docs (96x): 20 seconds...")
	shortThroughput := measureThroughput(rt, shortDocs, 96, 20)
	log.Printf("Scenario 4 - Throughput: %.1f emb/sec, Peak memory: %.0f MB",
		shortThroughput.Throughput, shortThroughput.PeakMemoryMB)

	// 6. Batch Long Docs (96x) - 20 seconds
	log.Printf("[6/6] Scenario 5 - Batch long docs (96x): 20 seconds...")
	longDocsRepeated := make([]string, 1)
	longDocsRepeated[0] = longDoc
	longThroughput := measureThroughput(rt, longDocsRepeated, 96, 20)
	log.Printf("Scenario 5 - Throughput: %.1f emb/sec, Peak memory: %.0f MB",
		longThroughput.Throughput, longThroughput.PeakMemoryMB)

	// Format and print results
	formatComprehensiveResults(platform, idleMemMB, shortLatency, longLatency,
		shortThroughput, longThroughput)
}
