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

	"github.com/headlands-org/go-semantica"
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
	embedDim     = flag.Int("dim", semantica.DefaultEmbedDim, "Embedding dimension (768, 512, 256, 128)")
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

// extraLongDoc replicates the long document to ensure sequence length comfortably
// exceeds the chunking threshold (>= ~300 tokens in practice).
var extraLongDoc = strings.TrimSpace(strings.Repeat(longDoc+" ", 8))

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
		fmt.Fprintf(os.Stderr, "        Benchmark duration in seconds (not required for comprehensive or single mode)\n")
		fmt.Fprintf(os.Stderr, "  -mode string\n")
		fmt.Fprintf(os.Stderr, "        Benchmark mode: 'batch', 'isolated', 'comprehensive', or 'single'\n")
		fmt.Fprintf(os.Stderr, "\nOptional flags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample commands:\n")
		fmt.Fprintf(os.Stderr, "  # Batch mode with 10 second run\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode batch -duration 10\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Isolated mode with custom workers and duration\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode isolated -duration 30 -workers 8\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Comprehensive benchmark for README documentation\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Single-document latency benchmark\n")
		fmt.Fprintf(os.Stderr, "  %s -model model/embeddinggemma-300m-Q8_0.gguf -mode single\n\n", os.Args[0])
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

	if *mode != "batch" && *mode != "isolated" && *mode != "comprehensive" && *mode != "single" {
		fmt.Fprintf(os.Stderr, "Error: -mode must be 'batch', 'isolated', 'comprehensive', or 'single'\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Duration is required for batch and isolated modes, but not comprehensive or single
	if (*mode == "batch" || *mode == "isolated") && *duration <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -duration must be greater than 0 for batch/isolated modes\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if _, err := semantica.ResolveDim(*embedDim); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n\n", err)
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

	// Handle single mode separately
	if *mode == "single" {
		runSingleMode()
		return
	}

	// Load model for batch/isolated modes
	log.Printf("Loading model from %s...", *modelPath)
	startLoad := time.Now()
	rt, err := semantica.Open(*modelPath)
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
	var computeDuration time.Duration
	var workerCounts []int

	log.Printf("Running %s mode benchmark for %d seconds...", *mode, *duration)
	log.Printf("Workers: %d, Batch size: %d", *workers, *batchSize)

	switch *mode {
	case "batch":
		embeddings, benchDuration, computeDuration = runBatchMode(rt)
	case "isolated":
		embeddings, benchDuration, computeDuration, workerCounts = runIsolatedMode(rt)
	}

	// Collect memory statistics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Report results to stderr
	fmt.Fprintf(os.Stderr, "\n=== Benchmark Results ===\n")
	fmt.Fprintf(os.Stderr, "Mode: %s\n", *mode)
	fmt.Fprintf(os.Stderr, "Duration: %v\n", benchDuration)
	fmt.Fprintf(os.Stderr, "Total embeddings: %d\n", embeddings)
	throughput := float64(embeddings) / benchDuration.Seconds()
	avgLatency := benchDuration / time.Duration(embeddings)
	computeSeconds := computeDuration.Seconds()
	computePerMs := 0.0
	if embeddings > 0 && computeSeconds > 0 {
		computePerMs = (computeSeconds * 1000.0) / float64(embeddings)
	}
	fmt.Fprintf(os.Stderr, "Throughput: %.2f embeddings/sec\n", throughput)
	fmt.Fprintf(os.Stderr, "Average latency: %v per embedding\n", avgLatency)
	fmt.Fprintf(os.Stderr, "Compute time: %.3fs (%.3f ms per embedding)\n", computeSeconds, computePerMs)

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
// Returns (total embeddings generated, actual wall duration, total CPU compute time)
func runBatchMode(rt *semantica.Runtime) (int, time.Duration, time.Duration) {
	startTime := time.Now()
	cpuStart := cpuTimeNow()
	deadline := startTime.Add(time.Duration(*duration) * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	computeReturn := func(total int) (int, time.Duration, time.Duration) {
		actual := time.Since(startTime)
		compute := cpuTimeNow() - cpuStart
		if compute < 0 {
			compute = 0
		}
		return total, actual, compute
	}

	totalEmbeddings := 0

	for {
		select {
		case <-ctx.Done():
			return computeReturn(totalEmbeddings)
		default:
		}

		inputs := make([]semantica.Input, *batchSize)
		for i := 0; i < *batchSize; i++ {
			inputs[i] = semantica.Input{
				Task:    semantica.TaskNone,
				Content: testTexts[rand.Intn(len(testTexts))],
				Dim:     *embedDim,
			}
		}

		_, err := rt.EmbedInputs(ctx, inputs)
		if err != nil {
			if ctx.Err() != nil {
				return computeReturn(totalEmbeddings)
			}
			log.Printf("Warning: embedding failed: %v", err)
			continue
		}

		totalEmbeddings += len(inputs)
	}
}

// workerResult contains results from a single worker
type workerResult struct {
	workerID int
	count    int
}

// runIsolatedMode runs the isolated benchmark mode
// Returns (total embeddings generated, actual duration, compute time, per-worker counts)
func runIsolatedMode(rt *semantica.Runtime) (int, time.Duration, time.Duration, []int) {
	// Note: rt parameter is unused in isolated mode - each worker loads its own model
	_ = rt

	// Set up timing and context
	startTime := time.Now()
	cpuStart := cpuTimeNow()
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
	computeDuration := cpuTimeNow() - cpuStart
	if computeDuration < 0 {
		computeDuration = 0
	}
	return totalEmbeddings, actualDuration, computeDuration, workerCounts
}

// worker is a goroutine that independently loads a model and generates embeddings
func worker(ctx context.Context, workerID int, wg *sync.WaitGroup, results chan<- workerResult) {
	defer wg.Done()

	// Each worker loads its own independent model instance
	rt, err := semantica.Open(*modelPath)
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

		// Generate single embedding without modifying the content.
		_, err := rt.EmbedSingleInput(ctx, semantica.Input{
			Task:    semantica.TaskNone,
			Content: text,
			Dim:     *embedDim,
		})
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
	Mean        float64
	P50         float64
	P95         float64
	P99         float64
	ComputeMean float64
	ComputeP50  float64
	ComputeP95  float64
	ComputeP99  float64
}

// ThroughputStats contains throughput statistics
type ThroughputStats struct {
	Throughput      float64
	PeakMemoryMB    float64
	Duration        float64
	TotalEmbeddings int
	ComputeSeconds  float64
	ComputePerMS    float64
}

// measureIdleMemory measures memory after model load with GC
func measureIdleMemory(rt *semantica.Runtime) uint64 {
	// Force GC
	runtime.GC()

	// Wait for stability
	time.Sleep(100 * time.Millisecond)

	// Read memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return m.HeapAlloc
}

func documentInput(content string) semantica.Input {
	return semantica.Input{
		Task:    semantica.TaskNone,
		Content: content,
		Dim:     *embedDim,
	}
}

// warmup runs warmup embeddings to warm up caches
func warmup(rt *semantica.Runtime, doc string) {
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_, _ = rt.EmbedSingleInput(ctx, documentInput(doc))
	}
}

func calculateStats(values []float64) (mean, p50, p95, p99 float64) {
	if len(values) == 0 {
		return 0, 0, 0, 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	mean = sum / float64(len(sorted))

	n := len(sorted)
	idx50 := int(float64(n) * 0.50)
	if idx50 >= n {
		idx50 = n - 1
	}
	idx95 := int(float64(n) * 0.95)
	if idx95 >= n {
		idx95 = n - 1
	}
	idx99 := int(float64(n) * 0.99)
	if idx99 >= n {
		idx99 = n - 1
	}

	p50 = sorted[idx50]
	p95 = sorted[idx95]
	p99 = sorted[idx99]
	return mean, p50, p95, p99
}

// measureSingleDocLatency measures single document latency with percentiles
func measureSingleDocLatency(rt *semantica.Runtime, doc string) LatencyStats {
	ctx := context.Background()
	const numRuns = 20

	latencies := make([]float64, numRuns)
	compute := make([]float64, numRuns)

	input := documentInput(doc)

	for i := 0; i < numRuns; i++ {
		cpuBefore := cpuTimeNow()
		start := time.Now()
		_, err := rt.EmbedSingleInput(ctx, input)
		cpuAfter := cpuTimeNow()
		if err != nil {
			log.Printf("Warning: embedding failed during latency test: %v", err)
			latencies[i] = 0
			compute[i] = 0
			continue
		}
		latencies[i] = float64(time.Since(start).Microseconds()) / 1000.0 // Convert to ms
		computeDur := cpuAfter - cpuBefore
		if computeDur < 0 {
			computeDur = 0
		}
		compute[i] = float64(computeDur.Microseconds()) / 1000.0
	}

	mean, p50, p95, p99 := calculateStats(latencies)
	cMean, c50, c95, c99 := calculateStats(compute)

	return LatencyStats{
		Mean:        mean,
		P50:         p50,
		P95:         p95,
		P99:         p99,
		ComputeMean: cMean,
		ComputeP50:  c50,
		ComputeP95:  c95,
		ComputeP99:  c99,
	}
}

// measureThroughput runs throughput test with memory sampling
// docs can be a slice of varied documents (for short doc test) or a single document repeated (for long doc test)
func measureThroughput(rt *semantica.Runtime, docs []string, batchSizeOverride int, durationSec int) ThroughputStats {
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
	cpuStart := cpuTimeNow()
	deadline := startTime.Add(time.Duration(durationSec) * time.Second)
	totalEmbeddings := 0

	for time.Now().Before(deadline) {
		// Select batch-size random texts from docs and build document prompts.
		inputs := make([]semantica.Input, batchSizeOverride)
		for i := 0; i < batchSizeOverride; i++ {
			inputs[i] = documentInput(docs[rand.Intn(len(docs))])
		}

		// Generate embeddings
		_, err := rt.EmbedInputs(ctx, inputs)
		if err != nil {
			log.Printf("Warning: embedding failed during throughput test: %v", err)
			continue
		}

		totalEmbeddings += len(inputs)
	}

	actualDuration := time.Since(startTime).Seconds()
	computeDuration := cpuTimeNow() - cpuStart
	if computeDuration < 0 {
		computeDuration = 0
	}
	computeSeconds := computeDuration.Seconds()

	// Stop memory sampling
	close(done)
	time.Sleep(150 * time.Millisecond) // Wait for goroutine to finish

	peak := atomic.LoadUint64(&peakMemory)

	computePer := 0.0
	if totalEmbeddings > 0 && computeSeconds > 0 {
		computePer = (computeSeconds * 1000.0) / float64(totalEmbeddings)
	}

	return ThroughputStats{
		Throughput:      float64(totalEmbeddings) / actualDuration,
		PeakMemoryMB:    float64(peak) / 1024 / 1024,
		Duration:        actualDuration,
		TotalEmbeddings: totalEmbeddings,
		ComputeSeconds:  computeSeconds,
		ComputePerMS:    computePer,
	}
}

// formatComprehensiveResults formats the comprehensive benchmark results
func formatComprehensiveResults(platform PlatformInfo, idleMemMB float64,
	shortLatency LatencyStats, longLatency LatencyStats, extraLongLatency LatencyStats,
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
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P50", shortLatency.ComputeP50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P95", shortLatency.ComputeP95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P99", shortLatency.ComputeP99, "ms")
	fmt.Printf("\n")

	// Scenario 3: Single Long Doc (49w)
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Single Long Doc (49w)", "P50 Latency", longLatency.P50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P95 Latency", longLatency.P95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P99 Latency", longLatency.P99, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P50", longLatency.ComputeP50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P95", longLatency.ComputeP95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P99", longLatency.ComputeP99, "ms")
	fmt.Printf("\n")

	// Scenario 4: Single Extra-Long Doc (~400w)
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Single Extra-Long Doc (~400w)", "P50 Latency", extraLongLatency.P50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P95 Latency", extraLongLatency.P95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "P99 Latency", extraLongLatency.P99, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P50", extraLongLatency.ComputeP50, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P95", extraLongLatency.ComputeP95, "ms")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "CPU P99", extraLongLatency.ComputeP99, "ms")
	fmt.Printf("\n")

	// Scenario 5: Batch Short Docs (96x)
	avgLatencyShort := 1000.0 / shortThroughput.Throughput // Convert to ms/emb
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Batch Short Docs (96x)", "Throughput", shortThroughput.Throughput, "emb/sec")
	fmt.Printf("%-32s%-20s%-12.0f%s\n", "", "Peak Memory", shortThroughput.PeakMemoryMB, "MB")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "Avg Latency", avgLatencyShort, "ms/emb")
	fmt.Printf("%-32s%-20s%-12.3f%s\n", "", "CPU Time", shortThroughput.ComputeSeconds, "s")
	fmt.Printf("%-32s%-20s%-12.3f%s\n", "", "CPU per Embedding", shortThroughput.ComputePerMS, "ms/emb")
	fmt.Printf("\n")

	// Scenario 6: Batch Long Docs (96x)
	avgLatencyLong := 1000.0 / longThroughput.Throughput // Convert to ms/emb
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "Batch Long Docs (96x)", "Throughput", longThroughput.Throughput, "emb/sec")
	fmt.Printf("%-32s%-20s%-12.0f%s\n", "", "Peak Memory", longThroughput.PeakMemoryMB, "MB")
	fmt.Printf("%-32s%-20s%-12.1f%s\n", "", "Avg Latency", avgLatencyLong, "ms/emb")
	fmt.Printf("%-32s%-20s%-12.3f%s\n", "", "CPU Time", longThroughput.ComputeSeconds, "s")
	fmt.Printf("%-32s%-20s%-12.3f%s\n", "", "CPU per Embedding", longThroughput.ComputePerMS, "ms/emb")
}

// runComprehensiveMode runs the comprehensive benchmark suite
func runComprehensiveMode() {
	log.Printf("Running 6-scenario benchmark suite...")

	// 1. Platform detection
	log.Printf("[1/7] Detecting platform...")
	platform := detectPlatform()
	log.Printf("Detected: %s, %d cores, %s/%s", platform.CPU, platform.Cores, platform.OS, platform.Arch)

	// 2. Load model and measure idle memory
	log.Printf("[2/7] Loading model and measuring idle memory...")
	rt, err := semantica.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()

	idleMemBytes := measureIdleMemory(rt)
	idleMemMB := float64(idleMemBytes) / 1024 / 1024
	log.Printf("Scenario 1 - Idle memory: %.0f MB", idleMemMB)

	// 3. Single Short Doc (9w) - Warmup + Measure
	log.Printf("[3/7] Scenario 2 - Single short doc (9w): warmup + 20 runs...")
	warmup(rt, shortDoc)
	shortLatency := measureSingleDocLatency(rt, shortDoc)
	log.Printf("Scenario 2 - Short doc P50: %.1f ms (CPU %.1f ms), P95: %.1f ms, P99: %.1f ms",
		shortLatency.P50, shortLatency.ComputeP50, shortLatency.P95, shortLatency.P99)

	// 4. Single Long Doc (49w) - Warmup + Measure
	log.Printf("[4/7] Scenario 3 - Single long doc (49w): warmup + 20 runs...")
	warmup(rt, longDoc)
	longLatency := measureSingleDocLatency(rt, longDoc)
	log.Printf("Scenario 3 - Long doc P50: %.1f ms (CPU %.1f ms), P95: %.1f ms, P99: %.1f ms",
		longLatency.P50, longLatency.ComputeP50, longLatency.P95, longLatency.P99)

	// 5. Single Extra-Long Doc (~400w) - Warmup + Measure
	log.Printf("[5/7] Scenario 4 - Single extra-long doc (~400w): warmup + 20 runs...")
	warmup(rt, extraLongDoc)
	extraLongLatency := measureSingleDocLatency(rt, extraLongDoc)
	log.Printf("Scenario 4 - Extra-long doc P50: %.1f ms (CPU %.1f ms), P95: %.1f ms, P99: %.1f ms",
		extraLongLatency.P50, extraLongLatency.ComputeP50, extraLongLatency.P95, extraLongLatency.P99)

	// 6. Batch Short Docs (96x) - 20 seconds
	log.Printf("[6/7] Scenario 5 - Batch short docs (96x): 20 seconds...")
	shortThroughput := measureThroughput(rt, shortDocs, 96, 20)
	log.Printf("Scenario 5 - Throughput: %.1f emb/sec, Peak memory: %.0f MB, CPU time: %.3fs",
		shortThroughput.Throughput, shortThroughput.PeakMemoryMB, shortThroughput.ComputeSeconds)

	// 7. Batch Long Docs (96x) - 20 seconds
	log.Printf("[7/7] Scenario 6 - Batch long docs (96x): 20 seconds...")
	longDocsRepeated := make([]string, 1)
	longDocsRepeated[0] = longDoc
	longThroughput := measureThroughput(rt, longDocsRepeated, 96, 20)
	log.Printf("Scenario 6 - Throughput: %.1f emb/sec, Peak memory: %.0f MB, CPU time: %.3fs",
		longThroughput.Throughput, longThroughput.PeakMemoryMB, longThroughput.ComputeSeconds)

	// Format and print results
	formatComprehensiveResults(platform, idleMemMB, shortLatency, longLatency, extraLongLatency,
		shortThroughput, longThroughput)
}

// runSingleMode runs the single-document latency benchmark mode
func runSingleMode() {
	log.Printf("Running single-document latency benchmark...")

	// 1. Load model
	log.Printf("[1/4] Loading model from %s...", *modelPath)
	startLoad := time.Now()
	rt, err := semantica.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer rt.Close()
	loadDuration := time.Since(startLoad)
	log.Printf("Model loaded in %v", loadDuration)
	log.Printf("Embedding dimension: %d", rt.EmbedDim())
	log.Printf("Max sequence length: %d", rt.MaxSeqLen())

	// 2. Measure short document latency
	log.Printf("[2/4] Measuring short document (9w) - warmup + 100 iterations...")
	warmup(rt, shortDoc)
	shortLatency := measureSingleDocLatencyPrecise(rt, shortDoc, 100)
	log.Printf("Short doc - P50: %.0f µs (CPU %.0f µs), P95: %.0f µs, P99: %.0f µs",
		shortLatency.P50, shortLatency.ComputeP50, shortLatency.P95, shortLatency.P99)

	// 3. Measure long document latency
	log.Printf("[3/4] Measuring long document (49w) - warmup + 100 iterations...")
	warmup(rt, longDoc)
	longLatency := measureSingleDocLatencyPrecise(rt, longDoc, 100)
	log.Printf("Long doc - P50: %.0f µs (CPU %.0f µs), P95: %.0f µs, P99: %.0f µs",
		longLatency.P50, longLatency.ComputeP50, longLatency.P95, longLatency.P99)

	// 4. Measure extra-long document latency
	log.Printf("[4/4] Measuring extra-long document (~400w) - warmup + 100 iterations...")
	warmup(rt, extraLongDoc)
	extraLongLatency := measureSingleDocLatencyPrecise(rt, extraLongDoc, 100)
	log.Printf("Extra-long doc - P50: %.0f µs (CPU %.0f µs), P95: %.0f µs, P99: %.0f µs",
		extraLongLatency.P50, extraLongLatency.ComputeP50, extraLongLatency.P95, extraLongLatency.P99)

	// Format and print results
	formatSingleModeResults(shortLatency, longLatency, extraLongLatency)
}

// measureSingleDocLatencyPrecise measures single document latency with microsecond precision
func measureSingleDocLatencyPrecise(rt *semantica.Runtime, doc string, numRuns int) LatencyStats {
	ctx := context.Background()

	latencies := make([]float64, numRuns)
	compute := make([]float64, numRuns)

	input := documentInput(doc)

	for i := 0; i < numRuns; i++ {
		cpuBefore := cpuTimeNow()
		start := time.Now()
		_, err := rt.EmbedSingleInput(ctx, input)
		cpuAfter := cpuTimeNow()
		if err != nil {
			log.Printf("Warning: embedding failed during latency test: %v", err)
			latencies[i] = 0
			compute[i] = 0
			continue
		}
		latencies[i] = float64(time.Since(start).Microseconds())
		computeDur := cpuAfter - cpuBefore
		if computeDur < 0 {
			computeDur = 0
		}
		compute[i] = float64(computeDur.Microseconds())
	}

	mean, p50, p95, p99 := calculateStats(latencies)
	cMean, c50, c95, c99 := calculateStats(compute)

	return LatencyStats{
		Mean:        mean,
		P50:         p50,
		P95:         p95,
		P99:         p99,
		ComputeMean: cMean,
		ComputeP50:  c50,
		ComputeP95:  c95,
		ComputeP99:  c99,
	}
}

// formatSingleModeResults formats the single mode benchmark results
func formatSingleModeResults(shortLatency, longLatency, extraLongLatency LatencyStats) {
	fmt.Printf("\n=== Single-Document Latency Benchmark Results ===\n\n")

	fmt.Printf("%-32s%-12s%-12s%-12s%-12s\n", "Document Type", "P50 (µs)", "P95 (µs)", "P99 (µs)", "Mean (µs)")
	fmt.Printf("--------------------------------------------------------------------------------\n")

	// Short document (9 words)
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Short (9w)",
		shortLatency.P50, shortLatency.P95, shortLatency.P99, shortLatency.Mean)

	// Long document (49 words)
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Long (49w)",
		longLatency.P50, longLatency.P95, longLatency.P99, longLatency.Mean)

	// Extra-long document (~400 words)
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Extra-Long (~400w)",
		extraLongLatency.P50, extraLongLatency.P95, extraLongLatency.P99, extraLongLatency.Mean)

	fmt.Printf("\n=== CPU Compute Time (µs) ===\n\n")
	fmt.Printf("%-32s%-12s%-12s%-12s%-12s\n", "Document Type", "P50", "P95", "P99", "Mean")
	fmt.Printf("--------------------------------------------------------------------------------\n")
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Short (9w)",
		shortLatency.ComputeP50, shortLatency.ComputeP95, shortLatency.ComputeP99, shortLatency.ComputeMean)
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Long (49w)",
		longLatency.ComputeP50, longLatency.ComputeP95, longLatency.ComputeP99, longLatency.ComputeMean)
	fmt.Printf("%-32s%-12.0f%-12.0f%-12.0f%-12.0f\n", "Extra-Long (~400w)",
		extraLongLatency.ComputeP50, extraLongLatency.ComputeP95, extraLongLatency.ComputeP99, extraLongLatency.ComputeMean)

	fmt.Printf("\n")

	// Calculate and display overhead analysis
	fmt.Printf("=== Latency Analysis ===\n\n")

	// Ratio of long to short
	longToShortRatio := longLatency.P50 / shortLatency.P50
	fmt.Printf("Long/Short ratio (P50):           %.2fx\n", longToShortRatio)

	// Ratio of extra-long to long
	extraLongToLongRatio := extraLongLatency.P50 / longLatency.P50
	fmt.Printf("Extra-Long/Long ratio (P50):      %.2fx\n", extraLongToLongRatio)

	// Calculate per-word latency estimates
	shortPerWord := shortLatency.P50 / 9.0
	longPerWord := longLatency.P50 / 49.0
	extraLongPerWord := extraLongLatency.P50 / 400.0

	fmt.Printf("\nPer-word latency estimates (P50):\n")
	fmt.Printf("  Short document:                 %.1f µs/word\n", shortPerWord)
	fmt.Printf("  Long document:                  %.1f µs/word\n", longPerWord)
	fmt.Printf("  Extra-Long document:            %.1f µs/word\n", extraLongPerWord)

	shortCpuPerWord := shortLatency.ComputeP50 / 9.0
	longCpuPerWord := longLatency.ComputeP50 / 49.0
	extraLongCpuPerWord := extraLongLatency.ComputeP50 / 400.0
	fmt.Printf("\nPer-word CPU compute (P50):\n")
	fmt.Printf("  Short document:                 %.1f µs/word\n", shortCpuPerWord)
	fmt.Printf("  Long document:                  %.1f µs/word\n", longCpuPerWord)
	fmt.Printf("  Extra-Long document:            %.1f µs/word\n", extraLongCpuPerWord)

	// Estimated fixed overhead
	// Assuming linear relationship: total_latency = fixed_overhead + (per_word_cost * num_words)
	// Using short and long documents to estimate:
	// shortLatency.P50 = overhead + (per_word * 9)
	// longLatency.P50 = overhead + (per_word * 49)
	// Solving: per_word = (longLatency.P50 - shortLatency.P50) / (49 - 9)
	estimatedPerWordCost := (longLatency.P50 - shortLatency.P50) / (49.0 - 9.0)
	estimatedOverhead := shortLatency.P50 - (estimatedPerWordCost * 9.0)

	fmt.Printf("\nEstimated overhead analysis:\n")
	fmt.Printf("  Fixed overhead:                 %.0f µs\n", estimatedOverhead)
	fmt.Printf("  Variable cost per word:         %.1f µs/word\n", estimatedPerWordCost)

	estimatedCpuPerWord := (longLatency.ComputeP50 - shortLatency.ComputeP50) / (49.0 - 9.0)
	estimatedCpuOverhead := shortLatency.ComputeP50 - (estimatedCpuPerWord * 9.0)
	fmt.Printf("\nEstimated CPU overhead analysis:\n")
	fmt.Printf("  Fixed CPU overhead:             %.0f µs\n", estimatedCpuOverhead)
	fmt.Printf("  Variable CPU cost per word:     %.1f µs/word\n", estimatedCpuPerWord)
	fmt.Printf("\n")
}
