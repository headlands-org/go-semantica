#!/bin/bash

# profile-optimization.sh - Profile both current and optimized configurations
# Compares fine-grained (DisableMatmulParallel=false) vs coarse-grained (DisableMatmulParallel=true)
# parallelism strategies.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_PATH="model/embeddinggemma-300m-Q8_0.gguf"
BATCH_SIZE=32
ITERATIONS=100
PROFILE_DIR="docs/profiles"
CURRENT_DIR="${PROFILE_DIR}/current"
OPTIMIZED_DIR="${PROFILE_DIR}/optimized"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Profiling Optimization: Fine-Grained vs Coarse-Grained${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Workers: NumCPU (auto-detected)"
echo "  Iterations: ${ITERATIONS}"
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}Error: Model file not found: ${MODEL_PATH}${NC}"
    echo "Please download the model first:"
    echo "  git lfs pull --include=\"model/embeddinggemma-300m-Q8_0.gguf\""
    exit 1
fi

# Build the profiling tool
echo -e "${GREEN}[1/5] Building profiling tool...${NC}"
go build -o /tmp/profile-runtime ./cmd/profile-runtime
echo "  Built: /tmp/profile-runtime"
echo ""

# Clean and create profile directories
rm -rf "${CURRENT_DIR}" "${OPTIMIZED_DIR}"
mkdir -p "${CURRENT_DIR}" "${OPTIMIZED_DIR}"

# Run Profile 1 - Current (Fine-Grained Parallelism)
echo -e "${GREEN}[2/5] Running CURRENT configuration (DisableMatmulParallel=false)...${NC}"
echo "  This uses fine-grained parallelism within matmul operations"
echo "  Expected: High goroutine scheduler overhead, many goroutines"
echo ""
/tmp/profile-runtime \
    -model="${MODEL_PATH}" \
    -batch="${BATCH_SIZE}" \
    -workers=0 \
    -disable-matmul-parallel=false \
    -iterations="${ITERATIONS}" \
    -cpuprofile="${CURRENT_DIR}/cpu.pprof" \
    -memprofile="${CURRENT_DIR}/mem.pprof" \
    2>&1 | tee "${CURRENT_DIR}/run.log"
echo ""

# Run Profile 2 - Optimized (Coarse-Grained Parallelism)
echo -e "${GREEN}[3/5] Running OPTIMIZED configuration (DisableMatmulParallel=true)...${NC}"
echo "  This uses coarse-grained parallelism across texts"
echo "  Expected: Lower goroutine overhead, fewer goroutines"
echo ""
/tmp/profile-runtime \
    -model="${MODEL_PATH}" \
    -batch="${BATCH_SIZE}" \
    -workers=0 \
    -disable-matmul-parallel=true \
    -iterations="${ITERATIONS}" \
    -cpuprofile="${OPTIMIZED_DIR}/cpu.pprof" \
    -memprofile="${OPTIMIZED_DIR}/mem.pprof" \
    2>&1 | tee "${OPTIMIZED_DIR}/run.log"
echo ""

# Generate analysis reports
echo -e "${GREEN}[4/5] Generating analysis reports...${NC}"

# Current profile analysis
echo "  Analyzing current configuration..."
go tool pprof -text -nodecount=30 "${CURRENT_DIR}/cpu.pprof" > "${CURRENT_DIR}/cpu_top30.txt"
go tool pprof -text -nodecount=30 "${CURRENT_DIR}/mem.pprof" > "${CURRENT_DIR}/mem_top30.txt"
go tool pprof -text -nodecount=30 -alloc_space "${CURRENT_DIR}/mem.pprof" > "${CURRENT_DIR}/mem_alloc_space.txt"
go tool pprof -text -nodecount=30 -alloc_objects "${CURRENT_DIR}/mem.pprof" > "${CURRENT_DIR}/mem_alloc_objects.txt"

# Extract key metrics from current
CURRENT_RUNTIME=$(grep "goroutine" "${CURRENT_DIR}/cpu_top30.txt" | head -1 | awk '{print $1}' || echo "0")
CURRENT_PROCYIELD=$(grep "procyield" "${CURRENT_DIR}/cpu_top30.txt" | head -1 | awk '{print $1}' || echo "0")
CURRENT_ALLOC_SPACE=$(head -1 "${CURRENT_DIR}/mem_alloc_space.txt" | awk '{print $5}' || echo "unknown")
CURRENT_ALLOC_OBJECTS=$(head -1 "${CURRENT_DIR}/mem_alloc_objects.txt" | awk '{print $5}' || echo "unknown")

# Optimized profile analysis
echo "  Analyzing optimized configuration..."
go tool pprof -text -nodecount=30 "${OPTIMIZED_DIR}/cpu.pprof" > "${OPTIMIZED_DIR}/cpu_top30.txt"
go tool pprof -text -nodecount=30 "${OPTIMIZED_DIR}/mem.pprof" > "${OPTIMIZED_DIR}/mem_top30.txt"
go tool pprof -text -nodecount=30 -alloc_space "${OPTIMIZED_DIR}/mem.pprof" > "${OPTIMIZED_DIR}/mem_alloc_space.txt"
go tool pprof -text -nodecount=30 -alloc_objects "${OPTIMIZED_DIR}/mem.pprof" > "${OPTIMIZED_DIR}/mem_alloc_objects.txt"

# Extract key metrics from optimized
OPTIMIZED_RUNTIME=$(grep "goroutine" "${OPTIMIZED_DIR}/cpu_top30.txt" | head -1 | awk '{print $1}' || echo "0")
OPTIMIZED_PROCYIELD=$(grep "procyield" "${OPTIMIZED_DIR}/cpu_top30.txt" | head -1 | awk '{print $1}' || echo "0")
OPTIMIZED_ALLOC_SPACE=$(head -1 "${OPTIMIZED_DIR}/mem_alloc_space.txt" | awk '{print $5}' || echo "unknown")
OPTIMIZED_ALLOC_OBJECTS=$(head -1 "${OPTIMIZED_DIR}/mem_alloc_objects.txt" | awk '{print $5}' || echo "unknown")

# Generate flame graphs if graphviz is available
if command -v dot &> /dev/null; then
    echo "  Generating flame graphs..."
    go tool pprof -svg "${CURRENT_DIR}/cpu.pprof" > "${CURRENT_DIR}/cpu_flame.svg" 2>/dev/null || true
    go tool pprof -svg "${OPTIMIZED_DIR}/cpu.pprof" > "${OPTIMIZED_DIR}/cpu_flame.svg" 2>/dev/null || true

    # Generate memory flame graphs
    go tool pprof -svg -alloc_space "${CURRENT_DIR}/mem.pprof" > "${CURRENT_DIR}/mem_alloc_flame.svg" 2>/dev/null || true
    go tool pprof -svg -alloc_space "${OPTIMIZED_DIR}/mem.pprof" > "${OPTIMIZED_DIR}/mem_alloc_flame.svg" 2>/dev/null || true
else
    echo -e "  ${YELLOW}Skipping flame graphs (graphviz not installed)${NC}"
    echo "  Install with: brew install graphviz"
fi

echo ""

# Extract timing data from logs
CURRENT_TIME=$(grep "Avg time per iteration:" "${CURRENT_DIR}/run.log" | awk '{print $5}')
OPTIMIZED_TIME=$(grep "Avg time per iteration:" "${OPTIMIZED_DIR}/run.log" | awk '{print $5}')

# Generate comparative analysis document
echo -e "${GREEN}[5/5] Creating analysis document...${NC}"

cat > "${PROFILE_DIR}/PROFILE_ANALYSIS.md" <<EOF
# CPU Profiling Analysis: Fine-Grained vs Coarse-Grained Parallelism

**Date:** $(date '+%Y-%m-%d %H:%M:%S')

## Executive Summary

This analysis compares two parallelism strategies for batch embedding generation:

1. **Current (Fine-Grained)**: \`DisableMatmulParallel=false\`
   - Parallelism occurs *within* matrix multiplication operations
   - Each text processed serially, but each matmul uses multiple goroutines

2. **Optimized (Coarse-Grained)**: \`DisableMatmulParallel=true\`
   - Parallelism occurs *across* texts in the batch
   - Multiple texts processed in parallel, each using serial matmul

## Test Configuration

- **Model**: \`${MODEL_PATH}\`
- **Batch Size**: ${BATCH_SIZE} texts
- **Workers**: NumCPU (auto-detected)
- **Iterations**: ${ITERATIONS}
- **Test Data**: Mix of short (5 tokens), medium (15 tokens), and long (40 tokens) texts

## Performance Results

### Timing

| Configuration | Avg Time/Iteration | Relative |
|--------------|-------------------|----------|
| Current (Fine-Grained) | ${CURRENT_TIME} | 1.00x |
| Optimized (Coarse-Grained) | ${OPTIMIZED_TIME} | TBD |

### CPU Profile Hotspots

#### Current (Fine-Grained)
\`\`\`
$(head -20 "${CURRENT_DIR}/cpu_top30.txt")
\`\`\`

#### Optimized (Coarse-Grained)
\`\`\`
$(head -20 "${OPTIMIZED_DIR}/cpu_top30.txt")
\`\`\`

### Memory Allocation

#### Allocation Space

| Configuration | Total Alloc Space |
|--------------|------------------|
| Current | ${CURRENT_ALLOC_SPACE} |
| Optimized | ${OPTIMIZED_ALLOC_SPACE} |

#### Allocation Objects

| Configuration | Total Alloc Objects |
|--------------|---------------------|
| Current | ${CURRENT_ALLOC_OBJECTS} |
| Optimized | ${OPTIMIZED_ALLOC_OBJECTS} |

## Analysis

### 1. Goroutine Scheduler Overhead

**Question**: Has goroutine scheduler overhead decreased?

**Current (Fine-Grained)**:
- \`runtime.goroutine\` overhead: ${CURRENT_RUNTIME}
- \`procyield\` overhead: ${CURRENT_PROCYIELD}

**Optimized (Coarse-Grained)**:
- \`runtime.goroutine\` overhead: ${OPTIMIZED_RUNTIME}
- \`procyield\` overhead: ${OPTIMIZED_PROCYIELD}

**Finding**: $([ "${OPTIMIZED_RUNTIME}" \< "${CURRENT_RUNTIME}" ] 2>/dev/null && echo "✅ Scheduler overhead DECREASED" || echo "⚠️  Review scheduler metrics")

### 2. Goroutine Creation

**Question**: Are there fewer goroutines being created?

See detailed analysis in CPU profiles. Look for:
- \`runtime.newproc\` calls
- \`runtime.goexit\` calls
- Number of goroutines in flight during profiling

**Current Profile**: \`${CURRENT_DIR}/cpu_top30.txt\`
**Optimized Profile**: \`${OPTIMIZED_DIR}/cpu_top30.txt\`

### 3. Allocation Rate

**Question**: Has allocation rate decreased?

**Finding**: $([ "${OPTIMIZED_ALLOC_OBJECTS}" \< "${CURRENT_ALLOC_OBJECTS}" ] 2>/dev/null && echo "✅ Allocation rate DECREASED" || echo "⚠️  Review allocation metrics")

See:
- \`${CURRENT_DIR}/mem_alloc_space.txt\` vs \`${OPTIMIZED_DIR}/mem_alloc_space.txt\`
- \`${CURRENT_DIR}/mem_alloc_objects.txt\` vs \`${OPTIMIZED_DIR}/mem_alloc_objects.txt\`

### 4. New Hotspots

**Question**: What are the new hotspots in the optimized version?

Review the top 30 functions in:
- \`${OPTIMIZED_DIR}/cpu_top30.txt\`

Key areas to examine:
- Is actual computation (matmul, attention) now dominating?
- Has runtime overhead moved down the profile?
- Are there any unexpected new bottlenecks?

### 5. Further Optimization Opportunities

**Question**: Are there further optimization opportunities?

Examine:
1. **Memory allocations**: Can we reduce allocations in hot paths?
2. **Data locality**: Are we cache-friendly?
3. **Algorithmic improvements**: Any redundant computation?
4. **SIMD usage**: Are we maximizing vector instructions?

## Flame Graph Analysis

Visual flame graphs available at:
- **Current CPU**: \`${CURRENT_DIR}/cpu_flame.svg\`
- **Optimized CPU**: \`${OPTIMIZED_DIR}/cpu_flame.svg\`
- **Current Memory**: \`${CURRENT_DIR}/mem_alloc_flame.svg\`
- **Optimized Memory**: \`${OPTIMIZED_DIR}/mem_alloc_flame.svg\`

Compare flame graph widths to see relative time spent in different code paths.

## Interactive Analysis Commands

\`\`\`bash
# View current CPU profile interactively
go tool pprof ${CURRENT_DIR}/cpu.pprof

# View optimized CPU profile interactively
go tool pprof ${OPTIMIZED_DIR}/cpu.pprof

# Compare profiles side-by-side
go tool pprof -http=:8080 ${CURRENT_DIR}/cpu.pprof
go tool pprof -http=:8081 ${OPTIMIZED_DIR}/cpu.pprof

# View memory profiles
go tool pprof ${CURRENT_DIR}/mem.pprof
go tool pprof ${OPTIMIZED_DIR}/mem.pprof

# List top goroutine-related functions
grep -E "(goroutine|procyield|newproc|goexit)" ${CURRENT_DIR}/cpu_top30.txt
grep -E "(goroutine|procyield|newproc|goexit)" ${OPTIMIZED_DIR}/cpu_top30.txt

# List top allocation sites
head -30 ${CURRENT_DIR}/mem_alloc_space.txt
head -30 ${OPTIMIZED_DIR}/mem_alloc_space.txt
\`\`\`

## Recommendations

Based on this analysis:

1. **If Optimized is Faster**:
   - Document the speedup ratio
   - Update default configuration to use coarse-grained parallelism for batch processing
   - Add guidance on when to use each mode

2. **If Current is Faster**:
   - Investigate why coarse-grained underperforms
   - Check for load balancing issues
   - Consider hybrid approach

3. **Next Steps**:
   - Profile with different batch sizes (8, 16, 64, 128)
   - Test on different hardware (different CPU counts)
   - Measure impact on different text length distributions

## Raw Data

All raw profiles and logs are available in:
- \`${CURRENT_DIR}/\`
- \`${OPTIMIZED_DIR}/\`
EOF

echo -e "${GREEN}Analysis document created: ${PROFILE_DIR}/PROFILE_ANALYSIS.md${NC}"
echo ""

# Print summary
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}                    Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Profiles captured:"
echo "  Current (Fine-Grained):"
echo "    - CPU: ${CURRENT_DIR}/cpu.pprof"
echo "    - Memory: ${CURRENT_DIR}/mem.pprof"
echo "    - Avg Time: ${CURRENT_TIME}"
echo ""
echo "  Optimized (Coarse-Grained):"
echo "    - CPU: ${OPTIMIZED_DIR}/cpu.pprof"
echo "    - Memory: ${OPTIMIZED_DIR}/mem.pprof"
echo "    - Avg Time: ${OPTIMIZED_TIME}"
echo ""
echo "Analysis document: ${PROFILE_DIR}/PROFILE_ANALYSIS.md"
echo ""
echo "Next steps:"
echo "  1. Review analysis: cat ${PROFILE_DIR}/PROFILE_ANALYSIS.md"
echo "  2. View flame graphs (if available)"
echo "  3. Interactive analysis: go tool pprof -http=:8080 ${CURRENT_DIR}/cpu.pprof"
echo ""
echo -e "${GREEN}Profiling complete!${NC}"
