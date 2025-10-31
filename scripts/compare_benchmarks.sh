#!/bin/bash
# compare_benchmarks.sh
# Runs both Go and C++ comprehensive benchmarks and compares results side-by-side
#
# Usage:
#   ./scripts/compare_benchmarks.sh [model_path]
#
# Example:
#   ./scripts/compare_benchmarks.sh model/embeddinggemma-300m-Q8_0.gguf

set -e

# Configuration
MODEL_PATH="${1:-model/embeddinggemma-300m-Q8_0.gguf}"
GO_BENCHMARK="./benchmark"
CPP_BENCHMARK="./benchmark_cpp/build/benchmark_cpp"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo "Usage: $0 [model_path]"
    echo ""
    echo "Runs both Go and C++ comprehensive benchmarks and compares results."
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to GGUF model file (default: model/embeddinggemma-300m-Q8_0.gguf)"
    echo ""
    echo "Example:"
    echo "  $0 model/embeddinggemma-300m-Q8_0.gguf"
    echo ""
}

# Check for help flags
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

echo -e "${CYAN}=== Benchmark Comparison Tool ===${NC}"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    echo ""
    echo "Please provide a valid model path:"
    echo "  $0 path/to/model.gguf"
    exit 1
fi

echo -e "${GREEN}Using model: $MODEL_PATH${NC}"
echo ""

# Check if Go benchmark exists, build if needed
if [ ! -f "$GO_BENCHMARK" ]; then
    echo -e "${YELLOW}Go benchmark not found, building...${NC}"
    go build -o benchmark ./cmd/benchmark || {
        echo -e "${RED}Error: Failed to build Go benchmark${NC}"
        exit 1
    }
    echo -e "${GREEN}Go benchmark built successfully${NC}"
    echo ""
fi

# Check if C++ benchmark exists, build if needed
if [ ! -f "$CPP_BENCHMARK" ]; then
    echo -e "${YELLOW}C++ benchmark not found, building...${NC}"
    cd benchmark_cpp && make && cd .. || {
        echo -e "${RED}Error: Failed to build C++ benchmark${NC}"
        exit 1
    }
    echo -e "${GREEN}C++ benchmark built successfully${NC}"
    echo ""
fi

# Create temporary files for outputs
GO_OUTPUT=$(mktemp)
CPP_OUTPUT=$(mktemp)

# Cleanup temporary files on exit
trap "rm -f $GO_OUTPUT $CPP_OUTPUT" EXIT

# Run Go benchmark
echo -e "${BLUE}=== Running Go Benchmark ===${NC}"
echo ""
if ! $GO_BENCHMARK -model="$MODEL_PATH" -mode=comprehensive > "$GO_OUTPUT" 2>&1; then
    echo -e "${RED}Error: Go benchmark failed${NC}"
    cat "$GO_OUTPUT"
    exit 1
fi
echo -e "${GREEN}Go benchmark completed successfully${NC}"
echo ""

# Run C++ benchmark
echo -e "${BLUE}=== Running C++ Benchmark ===${NC}"
echo ""
if ! $CPP_BENCHMARK -model="$MODEL_PATH" -mode=comprehensive > "$CPP_OUTPUT" 2>&1; then
    echo -e "${RED}Error: C++ benchmark failed${NC}"
    cat "$CPP_OUTPUT"
    exit 1
fi
echo -e "${GREEN}C++ benchmark completed successfully${NC}"
echo ""

# Parse results from Go output
parse_go_results() {
    local file="$1"

    # Extract platform info
    GO_PLATFORM=$(grep "^Platform:" "$file" | sed 's/Platform: //')

    # Extract Idle Memory (format: "Idle Memory    Heap Allocated    324    MB")
    GO_IDLE_MEM=$(grep "^Idle Memory" "$file" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')

    # Extract Single Short Doc latencies (format: "Single Short Doc (9w)    P50 Latency    17.5    ms")
    GO_SHORT_P50=$(grep "^Single Short Doc" "$file" -A 2 | grep "P50 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    GO_SHORT_P95=$(grep "^Single Short Doc" "$file" -A 2 | grep "P95 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    GO_SHORT_P99=$(grep "^Single Short Doc" "$file" -A 2 | grep "P99 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')

    # Extract Single Long Doc latencies
    GO_LONG_P50=$(grep "^Single Long Doc" "$file" -A 2 | grep "P50 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    GO_LONG_P95=$(grep "^Single Long Doc" "$file" -A 2 | grep "P95 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    GO_LONG_P99=$(grep "^Single Long Doc" "$file" -A 2 | grep "P99 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')

    # Extract Batch Short Docs metrics
    GO_BATCH_SHORT_THROUGHPUT=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Throughput" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
    GO_BATCH_SHORT_MEM=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Peak Memory" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')
    GO_BATCH_SHORT_LATENCY=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Avg Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')

    # Extract Batch Long Docs metrics
    GO_BATCH_LONG_THROUGHPUT=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Throughput" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
    GO_BATCH_LONG_MEM=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Peak Memory" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')
    GO_BATCH_LONG_LATENCY=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Avg Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
}

# Parse results from C++ output
parse_cpp_results() {
    local file="$1"

    # Extract platform info
    CPP_PLATFORM=$(grep "^Platform:" "$file" | sed 's/Platform: //')

    # Extract Idle Memory (format: "Idle Memory    Heap Allocated    298    MB")
    CPP_IDLE_MEM=$(grep "^Idle Memory" "$file" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')

    # Extract Single Short Doc latencies (format: "Single Short Doc (9w)    P50 Latency    5.2    ms")
    CPP_SHORT_P50=$(grep "^Single Short Doc" "$file" -A 2 | grep "P50 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    CPP_SHORT_P95=$(grep "^Single Short Doc" "$file" -A 2 | grep "P95 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    CPP_SHORT_P99=$(grep "^Single Short Doc" "$file" -A 2 | grep "P99 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')

    # Extract Single Long Doc latencies
    CPP_LONG_P50=$(grep "^Single Long Doc" "$file" -A 2 | grep "P50 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    CPP_LONG_P95=$(grep "^Single Long Doc" "$file" -A 2 | grep "P95 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')
    CPP_LONG_P99=$(grep "^Single Long Doc" "$file" -A 2 | grep "P99 Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms"){print $i; exit}}}')

    # Extract Batch Short Docs metrics
    CPP_BATCH_SHORT_THROUGHPUT=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Throughput" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
    CPP_BATCH_SHORT_MEM=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Peak Memory" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')
    CPP_BATCH_SHORT_LATENCY=$(grep "^Batch Short Docs" "$file" -A 2 | grep "Avg Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')

    # Extract Batch Long Docs metrics
    CPP_BATCH_LONG_THROUGHPUT=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Throughput" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
    CPP_BATCH_LONG_MEM=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Peak Memory" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="MB"){print $i; exit}}}')
    CPP_BATCH_LONG_LATENCY=$(grep "^Batch Long Docs" "$file" -A 2 | grep "Avg Latency" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?$/){print $i; exit}}}')
}

# Calculate speedup factor (positive means C++ is faster, negative means Go is faster)
calculate_speedup() {
    local go_val="$1"
    local cpp_val="$2"

    # For latency metrics (lower is better), speedup = go/cpp
    # For throughput metrics (higher is better), speedup = cpp/go
    # We'll determine this based on context in the display function

    # Return the ratio as a string for awk to process
    echo "$go_val $cpp_val"
}

# Parse both outputs
echo -e "${YELLOW}Parsing benchmark results...${NC}"
parse_go_results "$GO_OUTPUT"
parse_cpp_results "$CPP_OUTPUT"
echo ""

# Display comparison table
echo -e "${CYAN}=== Benchmark Comparison: Go vs C++ ===${NC}"
echo ""
echo -e "${BLUE}Platform Info:${NC}"
echo "  Go:  $GO_PLATFORM"
echo "  C++: $CPP_PLATFORM"
echo ""

printf "%-35s%-15s%-15s%-15s\n" "Scenario / Metric" "Go" "C++" "Speedup"
printf "%s\n" "--------------------------------------------------------------------------------"

# Scenario 1: Idle Memory
printf "%-35s%-15s%-15s%-15s\n" "Idle Memory (MB)" "$GO_IDLE_MEM" "$CPP_IDLE_MEM" \
    "$(echo "$GO_IDLE_MEM $CPP_IDLE_MEM" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "\n"

# Scenario 2: Single Short Doc
printf "%-35s%-15s%-15s%-15s\n" "Single Short Doc (9w) - P50 (ms)" "$GO_SHORT_P50" "$CPP_SHORT_P50" \
    "$(echo "$GO_SHORT_P50 $CPP_SHORT_P50" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Single Short Doc (9w) - P95 (ms)" "$GO_SHORT_P95" "$CPP_SHORT_P95" \
    "$(echo "$GO_SHORT_P95 $CPP_SHORT_P95" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Single Short Doc (9w) - P99 (ms)" "$GO_SHORT_P99" "$CPP_SHORT_P99" \
    "$(echo "$GO_SHORT_P99 $CPP_SHORT_P99" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "\n"

# Scenario 3: Single Long Doc
printf "%-35s%-15s%-15s%-15s\n" "Single Long Doc (49w) - P50 (ms)" "$GO_LONG_P50" "$CPP_LONG_P50" \
    "$(echo "$GO_LONG_P50 $CPP_LONG_P50" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Single Long Doc (49w) - P95 (ms)" "$GO_LONG_P95" "$CPP_LONG_P95" \
    "$(echo "$GO_LONG_P95 $CPP_LONG_P95" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Single Long Doc (49w) - P99 (ms)" "$GO_LONG_P99" "$CPP_LONG_P99" \
    "$(echo "$GO_LONG_P99 $CPP_LONG_P99" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "\n"

# Scenario 4: Batch Short Docs
printf "%-35s%-15s%-15s%-15s\n" "Batch Short (96x) - Throughput" "$GO_BATCH_SHORT_THROUGHPUT" "$CPP_BATCH_SHORT_THROUGHPUT" \
    "$(echo "$GO_BATCH_SHORT_THROUGHPUT $CPP_BATCH_SHORT_THROUGHPUT" | awk '{if($1>0) printf "%.2fx", $2/$1; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Batch Short (96x) - Peak Mem (MB)" "$GO_BATCH_SHORT_MEM" "$CPP_BATCH_SHORT_MEM" \
    "$(echo "$GO_BATCH_SHORT_MEM $CPP_BATCH_SHORT_MEM" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Batch Short (96x) - Avg Lat (ms)" "$GO_BATCH_SHORT_LATENCY" "$CPP_BATCH_SHORT_LATENCY" \
    "$(echo "$GO_BATCH_SHORT_LATENCY $CPP_BATCH_SHORT_LATENCY" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "\n"

# Scenario 5: Batch Long Docs
printf "%-35s%-15s%-15s%-15s\n" "Batch Long (96x) - Throughput" "$GO_BATCH_LONG_THROUGHPUT" "$CPP_BATCH_LONG_THROUGHPUT" \
    "$(echo "$GO_BATCH_LONG_THROUGHPUT $CPP_BATCH_LONG_THROUGHPUT" | awk '{if($1>0) printf "%.2fx", $2/$1; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Batch Long (96x) - Peak Mem (MB)" "$GO_BATCH_LONG_MEM" "$CPP_BATCH_LONG_MEM" \
    "$(echo "$GO_BATCH_LONG_MEM $CPP_BATCH_LONG_MEM" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
printf "%-35s%-15s%-15s%-15s\n" "Batch Long (96x) - Avg Lat (ms)" "$GO_BATCH_LONG_LATENCY" "$CPP_BATCH_LONG_LATENCY" \
    "$(echo "$GO_BATCH_LONG_LATENCY $CPP_BATCH_LONG_LATENCY" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"

echo ""
echo -e "${GREEN}=== Summary ===${NC}"
echo ""
echo "Speedup factors indicate relative performance:"
echo "  - For latency/memory metrics: >1.0x means C++ is faster/uses less memory"
echo "  - For throughput metrics: >1.0x means C++ has higher throughput"
echo ""
echo "Key takeaways:"
echo "  - Single doc latency speedup: $(echo "$GO_SHORT_P50 $CPP_SHORT_P50" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
echo "  - Batch throughput speedup: $(echo "$GO_BATCH_SHORT_THROUGHPUT $CPP_BATCH_SHORT_THROUGHPUT" | awk '{if($1>0) printf "%.2fx", $2/$1; else print "N/A"}')"
echo "  - Memory efficiency: $(echo "$GO_IDLE_MEM $CPP_IDLE_MEM" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')"
echo ""

# Print detailed outputs if requested
if [ "${SHOW_DETAILS:-0}" == "1" ]; then
    echo -e "${YELLOW}=== Go Benchmark Output ===${NC}"
    cat "$GO_OUTPUT"
    echo ""
    echo -e "${YELLOW}=== C++ Benchmark Output ===${NC}"
    cat "$CPP_OUTPUT"
    echo ""
fi

echo -e "${GREEN}Comparison complete!${NC}"
exit 0
