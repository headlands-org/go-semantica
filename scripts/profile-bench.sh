#!/bin/bash

# profile-bench.sh - Profile Go benchmarks and generate visualization
# Usage: ./scripts/profile-bench.sh <benchmark-name> [package-path]
#
# Examples:
#   ./scripts/profile-bench.sh BenchmarkMatMul ./internal/kernels
#   ./scripts/profile-bench.sh BenchmarkEmbed ./internal/runtime
#   ./scripts/profile-bench.sh BenchmarkTokenize ./internal/tokenizer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 <benchmark-name> [package-path]"
    echo ""
    echo "Arguments:"
    echo "  benchmark-name  Name of the benchmark to run (e.g., BenchmarkMatMul)"
    echo "  package-path    Optional package path (default: ./internal/runtime)"
    echo ""
    echo "Examples:"
    echo "  $0 BenchmarkMatMul ./internal/kernels"
    echo "  $0 BenchmarkEmbed ./internal/runtime"
    echo "  $0 BenchmarkTokenize ./internal/tokenizer"
    echo ""
    echo "Output:"
    echo "  profiles/YYYY-MM-DD_HHMMSS_<benchmark-name>/cpu.pprof"
    echo "  profiles/YYYY-MM-DD_HHMMSS_<benchmark-name>/mem.pprof"
    echo "  profiles/YYYY-MM-DD_HHMMSS_<benchmark-name>/flame.svg (if -flame flag used)"
    exit 1
}

# Check for help flag first
if [ "$#" -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
    usage
fi

BENCHMARK_NAME=$1
PACKAGE_PATH=${2:-"./internal/runtime"}
GENERATE_FLAME=false

# Parse flags
shift
for arg in "$@"; do
    case $arg in
        -flame|--flame)
            GENERATE_FLAME=true
            ;;
        ./*)
            # Package path already set
            ;;
    esac
done

# Create profiles directory with timestamp
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
PROFILE_DIR="profiles/${TIMESTAMP}_${BENCHMARK_NAME}"
mkdir -p "$PROFILE_DIR"

echo -e "${GREEN}[1/4] Running benchmark: ${BENCHMARK_NAME}${NC}"
echo "Package: ${PACKAGE_PATH}"
echo "Output: ${PROFILE_DIR}"
echo ""

# Run benchmark with profiling
# Note: Use longer benchtime to collect meaningful CPU samples
go test -bench="^${BENCHMARK_NAME}$" \
    -cpuprofile="${PROFILE_DIR}/cpu.pprof" \
    -memprofile="${PROFILE_DIR}/mem.pprof" \
    -benchmem \
    -benchtime=5s \
    "${PACKAGE_PATH}"

# Check if profiles were generated
if [ ! -f "${PROFILE_DIR}/cpu.pprof" ]; then
    echo -e "${RED}Error: CPU profile not generated${NC}"
    exit 1
fi

if [ ! -f "${PROFILE_DIR}/mem.pprof" ]; then
    echo -e "${RED}Error: Memory profile not generated${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[2/4] Profiles generated successfully${NC}"
echo "  CPU: ${PROFILE_DIR}/cpu.pprof"
echo "  Memory: ${PROFILE_DIR}/mem.pprof"

# Generate text reports
echo ""
echo -e "${GREEN}[3/4] Generating text reports${NC}"

echo "  CPU top 20 functions..."
go tool pprof -text -nodecount=20 "${PROFILE_DIR}/cpu.pprof" > "${PROFILE_DIR}/cpu_top20.txt"

echo "  Memory top 20 allocations..."
go tool pprof -text -nodecount=20 "${PROFILE_DIR}/mem.pprof" > "${PROFILE_DIR}/mem_top20.txt"

echo "  Memory allocations (alloc_space)..."
go tool pprof -text -nodecount=20 -alloc_space "${PROFILE_DIR}/mem.pprof" > "${PROFILE_DIR}/mem_alloc_space.txt"

echo "  Memory in-use (inuse_space)..."
go tool pprof -text -nodecount=20 -inuse_space "${PROFILE_DIR}/mem.pprof" > "${PROFILE_DIR}/mem_inuse_space.txt"

# Generate flame graph if requested
if [ "$GENERATE_FLAME" = true ]; then
    echo ""
    echo -e "${GREEN}[4/4] Generating flame graph${NC}"

    # Check if graphviz is installed (required for SVG output)
    if command -v dot &> /dev/null; then
        echo "  Generating CPU flame graph..."
        go tool pprof -svg "${PROFILE_DIR}/cpu.pprof" > "${PROFILE_DIR}/flame.svg" 2>/dev/null
        if [ -s "${PROFILE_DIR}/flame.svg" ]; then
            echo "  Flame graph: ${PROFILE_DIR}/flame.svg"
        else
            rm -f "${PROFILE_DIR}/flame.svg"
            echo -e "${YELLOW}  Warning: No CPU samples collected (benchmark may be too short)${NC}"
        fi
    else
        echo -e "${YELLOW}  Warning: Graphviz not installed. Install with: brew install graphviz${NC}"
        echo "  Alternative: Use web UI instead:"
        echo "    go tool pprof -http=:8080 ${PROFILE_DIR}/cpu.pprof"
    fi
else
    echo ""
    echo -e "${YELLOW}[4/4] Skipping flame graph (use -flame flag to generate)${NC}"
fi

# Print summary
echo ""
echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo ""
echo "Profile directory: ${PROFILE_DIR}/"
echo ""
echo "Quick analysis commands:"
echo "  # Interactive CPU profile"
echo "  go tool pprof ${PROFILE_DIR}/cpu.pprof"
echo ""
echo "  # Interactive memory profile"
echo "  go tool pprof ${PROFILE_DIR}/mem.pprof"
echo ""
echo "  # View top CPU consumers"
echo "  cat ${PROFILE_DIR}/cpu_top20.txt"
echo ""
echo "  # View top memory allocations"
echo "  cat ${PROFILE_DIR}/mem_top20.txt"
echo ""
echo "  # Generate flame graph"
echo "  go tool pprof -svg ${PROFILE_DIR}/cpu.pprof > ${PROFILE_DIR}/flame.svg"
echo ""
echo "  # Web UI (interactive)"
echo "  go tool pprof -http=:8080 ${PROFILE_DIR}/cpu.pprof"
echo ""
