#!/bin/bash
# benchmark_llamacpp.sh
# Runs llama.cpp embedding benchmarks matching our 5-scenario matrix

set -e

# Configuration
MODEL_PATH="${1:-model/embeddinggemma-300m-Q8_0.gguf}"
LLAMACPP_DIR="${2:-../llama.cpp}"

# Check for llama.cpp embedding binary (system-wide or local build)
if command -v llama-embedding &> /dev/null; then
    EMBEDDING_BIN="llama-embedding"
elif [ -f "$LLAMACPP_DIR/embedding" ]; then
    EMBEDDING_BIN="$LLAMACPP_DIR/embedding"
else
    EMBEDDING_BIN=""
fi

# Test documents (must match cmd/benchmark/main.go exactly)
SHORT_DOC="The quick brown fox jumps over the lazy dog"
LONG_DOC="Machine learning has revolutionized artificial intelligence by enabling computers to learn from data without explicit programming. Modern neural networks can process vast amounts of information and identify complex patterns that would be impossible for humans to detect manually. This technology powers applications from image recognition to natural language processing."

# Short docs for batch test
declare -a SHORT_DOCS=(
    "The quick brown fox jumps over the lazy dog"
    "Artificial intelligence is transforming modern technology"
    "Machine learning enables computers to learn from data"
    "Neural networks process information efficiently"
    "Deep learning powers many AI applications"
)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Llama.cpp Benchmark Runner ==="
echo ""

# Check if llama.cpp embedding binary exists
if [ -z "$EMBEDDING_BIN" ]; then
    echo -e "${RED}Error: llama.cpp embedding binary not found${NC}"
    echo ""
    echo "To install llama.cpp (Arch Linux):"
    echo "  yay -S llama.cpp"
    echo ""
    echo "Or to build from source:"
    echo "  git clone https://github.com/ggerganov/llama.cpp $LLAMACPP_DIR"
    echo "  cd $LLAMACPP_DIR"
    echo "  make -j"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using llama.cpp: $EMBEDDING_BIN${NC}"
echo -e "${GREEN}Using model: $MODEL_PATH${NC}"
echo ""

# Detect platform
CPU_INFO=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU")
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "Platform: $CPU_INFO, $NUM_CORES cores, $OS/$ARCH"
echo ""

# Helper function to measure single document latency
measure_single_latency() {
    local doc="$1"
    local label="$2"
    local num_runs=20

    echo -e "${YELLOW}[$label] Running warmup (5 runs)...${NC}"
    for i in {1..5}; do
        $EMBEDDING_BIN -m "$MODEL_PATH" -p "$doc" --embd-normalize 2 --log-disable > /dev/null 2>&1
    done

    echo -e "${YELLOW}[$label] Measuring latency ($num_runs runs)...${NC}"

    # Run and collect timings
    latencies=()
    for i in $(seq 1 $num_runs); do
        timing=$( (time -p $EMBEDDING_BIN -m "$MODEL_PATH" -p "$doc" --embd-normalize 2 --log-disable > /dev/null 2>&1) 2>&1 | grep real | awk '{print $2}')
        latencies+=($timing)
    done

    # Sort and calculate percentiles
    sorted=($(printf '%s\n' "${latencies[@]}" | sort -n))

    # Convert to milliseconds and calculate percentiles
    p50_idx=$((num_runs / 2))
    p95_idx=$((num_runs * 95 / 100))
    p99_idx=$((num_runs * 99 / 100))

    p50=$(echo "${sorted[$p50_idx]} * 1000" | bc -l)
    p95=$(echo "${sorted[$p95_idx]} * 1000" | bc -l)
    p99=$(echo "${sorted[$p99_idx]} * 1000" | bc -l)

    printf "  P50: %.1f ms\n" $p50
    printf "  P95: %.1f ms\n" $p95
    printf "  P99: %.1f ms\n" $p99
    echo ""
}

# Helper function to measure idle memory
measure_idle_memory() {
    echo -e "${YELLOW}[Idle Memory] Starting llama.cpp and measuring RSS...${NC}"

    # Start embedding process and get its PID
    $EMBEDDING_BIN -m "$MODEL_PATH" -p "warmup" --embd-normalize 2 --log-disable > /dev/null 2>&1 &
    PID=$!

    # Wait for model to load
    sleep 2

    # Measure RSS
    if command -v ps &> /dev/null; then
        RSS=$(ps -p $PID -o rss= 2>/dev/null || echo "0")
        RSS_MB=$((RSS / 1024))
    else
        RSS_MB="Unknown"
    fi

    # Clean up
    kill $PID 2>/dev/null || true

    echo "  Heap Allocated: $RSS_MB MB (RSS)"
    echo ""
}

# Scenario 1: Idle Memory
echo "=== Scenario 1: Idle Memory ==="
measure_idle_memory

# Scenario 2: Single Short Doc (9w)
echo "=== Scenario 2: Single Short Doc (9w) ==="
measure_single_latency "$SHORT_DOC" "Single Short"

# Scenario 3: Single Long Doc (49w)
echo "=== Scenario 3: Single Long Doc (49w) ==="
measure_single_latency "$LONG_DOC" "Single Long"

# Scenario 4 & 5: Batch tests
echo "=== Scenarios 4 & 5: Batch Throughput ==="
echo -e "${YELLOW}Note: llama.cpp's embedding tool doesn't support batch processing${NC}"
echo -e "${YELLOW}We'll simulate by running multiple single embeddings${NC}"
echo ""
echo -e "${RED}Skipping batch scenarios - llama.cpp doesn't have equivalent batch API${NC}"
echo ""

# Summary
echo "=== Llama.cpp Benchmark Complete ==="
echo ""
echo "Note: For batch scenarios, llama.cpp would need a custom harness"
echo "that calls the embedding API multiple times in parallel."
echo ""
echo "Comparison with pure-go-llamas:"
echo "  Run: ./benchmark -model=$MODEL_PATH -mode=comprehensive"
