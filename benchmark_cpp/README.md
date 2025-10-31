# C++ Benchmark using llama.cpp

This directory contains a C++ benchmark for embedding generation using the llama.cpp library.

## Prerequisites

### Required Software

1. **C++17 or later compiler**
   - GCC 7+ or Clang 5+
   - On Ubuntu/Debian: `sudo apt install build-essential`
   - On macOS: `xcode-select --install`

2. **llama.cpp library** (latest main branch recommended)
   - Shared library (`libllama.so` on Linux, `libllama.dylib` on macOS) OR
   - Static library (`libllama.a`)
   - Header file (`llama.h`)

3. **Compatible GGUF model file**
   - Example: `embeddinggemma-300m-Q8_0.gguf`
   - Available at: `ggml-org/embeddinggemma-300M-GGUF` on HuggingFace

### Installing llama.cpp

#### Option 1: Build from Source (Recommended)

```bash
# Clone llama.cpp repository
cd /path/to/your/projects
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)  # Use -j$(sysctl -n hw.ncpu) on macOS

# Verify libraries exist
ls -l build/bin/libllama.*  # On Linux: libllama.so, on macOS: libllama.dylib
```

#### Option 2: Homebrew (macOS only)

```bash
brew install llama.cpp

# Note: Homebrew installation may have different library locations
# Set LLAMA_CPP_PATH accordingly (e.g., /opt/homebrew/Cellar/llama.cpp/...)
```

### Verifying llama.cpp Installation

```bash
# Check if header exists
find /path/to/llama.cpp -name "llama.h"
# Expected: /path/to/llama.cpp/include/llama.h or /path/to/llama.cpp/llama.h

# Check if libraries exist
find /path/to/llama.cpp -name "libllama.*"
# Expected: /path/to/llama.cpp/build/bin/libllama.so (or .dylib on macOS)
```

## Building

### Quick Start

```bash
# Navigate to benchmark_cpp directory
cd benchmark_cpp

# If llama.cpp is at ../llama.cpp (default)
make

# If llama.cpp is elsewhere, set LLAMA_CPP_PATH
export LLAMA_CPP_PATH=/home/user/llama.cpp
make

# Or specify inline
LLAMA_CPP_PATH=/opt/llama.cpp make
```

### Build Targets

- **`make all`** or **`make release`** - Optimized build with `-O3 -march=native` (default)
- **`make debug`** - Debug build with `-g -O0` (no optimization)
- **`make clean`** - Remove build artifacts
- **`make info`** - Display build configuration (useful for troubleshooting)
- **`make help`** - Show all available targets and usage

### Platform-Specific Build Instructions

#### Linux

```bash
cd benchmark_cpp

# Install build tools if needed
sudo apt install build-essential cmake  # Ubuntu/Debian
# or
sudo dnf install gcc-c++ cmake  # Fedora/RHEL

# Build llama.cpp first (if not already built)
cd ../llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd ../benchmark_cpp

# Build benchmark
make

# Verify build
./build/benchmark_cpp --help
```

#### macOS

```bash
cd benchmark_cpp

# Install Xcode command line tools if needed
xcode-select --install

# Build llama.cpp first (if not already built)
cd ../llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
cd ../benchmark_cpp

# Build benchmark
make

# Verify build
./build/benchmark_cpp --help
```

## Usage

### Command-Line Interface

```bash
./build/benchmark_cpp [options]

Required flags:
  -model string
        Path to GGUF model file
  -mode string
        Benchmark mode: 'batch', 'isolated', or 'comprehensive'
  -duration int
        Benchmark duration in seconds (required for batch/isolated modes)

Optional flags:
  -batch-size int
        Batch size for batch mode (default: NumCPU * 4)
  -workers int
        Number of worker threads (default: NumCPU * 2)
```

### Benchmark Modes

#### 1. Batch Mode

Tests maximum throughput with batched requests.

```bash
# Run batch mode with default settings (10 seconds)
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode batch -duration 10

# Custom batch size
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode batch -duration 10 -batch-size 64

# Longer run for more stable measurements
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode batch -duration 60
```

#### 2. Isolated Mode

Tests single-request latency with multiple concurrent workers.

```bash
# Run isolated mode with default workers (10 seconds)
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode isolated -duration 10

# Custom worker count (8 threads)
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode isolated -duration 10 -workers 8

# High concurrency test
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode isolated -duration 30 -workers 16
```

#### 3. Comprehensive Mode

Runs a standardized 5-scenario benchmark matrix.

```bash
# Run comprehensive benchmark (no duration needed)
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode comprehensive
```

**Scenarios Tested:**

1. **Short text, batch=1**: Single short document (9 words) - minimum latency
2. **Short text, batch=5**: Small batch of varied short documents (6-9 words)
3. **Long text, batch=1**: Single long document (49 words)
4. **Long text, batch=5**: Batch of long documents
5. **Mixed workload**: Alternating short/long texts

## Output Format

### Batch Mode Output

```
=== llama.cpp Batch Mode ===
Platform: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz (8 cores)
Model: ../model/embeddinggemma-300m-Q8_0.gguf
Duration: 10.00s
Batch Size: 32

Results:
  Throughput: 45.3 embeddings/sec
  Avg Latency: 22.1ms
  Memory: 892 MB (peak)
  CPU Utilization: 87.2%
```

### Isolated Mode Output

```
=== llama.cpp Isolated Mode ===
Platform: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz (8 cores)
Model: ../model/embeddinggemma-300m-Q8_0.gguf
Duration: 10.00s
Workers: 8

Results:
  Total Requests: 543
  Throughput: 54.3 embeddings/sec
  Latency:
    P50: 17.8ms
    P95: 34.2ms
    P99: 48.5ms
  Memory: 945 MB (peak)
  CPU Utilization: 91.4%
```

### Comprehensive Mode Output

```
=== llama.cpp Comprehensive Benchmark ===
Platform: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz (8 cores)
Model: ../model/embeddinggemma-300m-Q8_0.gguf

Scenario 1: Short text (9 words), batch=1
  Throughput: 62.1 embeddings/sec
  P50 Latency: 16.1ms
  P95 Latency: 18.3ms
  P99 Latency: 19.8ms

Scenario 2: Short batch (5 docs, 6-9 words each)
  Throughput: 178.5 embeddings/sec
  P50 Latency: 28.0ms
  P95 Latency: 31.2ms
  P99 Latency: 33.5ms

[... scenarios 3-5 ...]

=== Summary ===
  Average Throughput: 125.3 embeddings/sec
  Average P50 Latency: 24.5ms
  Peak Memory: 1024 MB
```

### Metrics Explained

- **Throughput**: Embeddings processed per second (higher is better)
- **Latency**: Time to process a single request (lower is better)
  - **P50**: 50th percentile (median) - half of requests are faster
  - **P95**: 95th percentile - 95% of requests are faster
  - **P99**: 99th percentile - 99% of requests are faster
- **Memory**: Peak RSS (Resident Set Size) in MB
- **CPU Utilization**: Average CPU usage across all cores

## Troubleshooting

### Build Issues

#### Error: `llama.cpp not found`

```bash
# Solution: Set LLAMA_CPP_PATH to correct location
export LLAMA_CPP_PATH=/path/to/your/llama.cpp
make info  # Verify detection
make clean && make
```

#### Error: `No llama.cpp library found`

```bash
# Solution: Build llama.cpp first
cd $LLAMA_CPP_PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify libraries were created
ls -l build/bin/libllama.*
# Expected: libllama.so (Linux) or libllama.dylib (macOS)

# Return to benchmark and rebuild
cd -
make clean && make
```

#### Error: `llama.h not found`

```bash
# Solution: Verify header location
find $LLAMA_CPP_PATH -name "llama.h"
# Should be in include/ or root directory

# If missing, ensure llama.cpp is up to date
cd $LLAMA_CPP_PATH
git pull origin main
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### Error: `cannot find -lllama`

```bash
# This means the library file exists but linker can't find it
# Solution: Rebuild with verbose output
make clean
make info  # Check detected library path
make V=1   # Verbose build to see exact linker command

# If library path is wrong, set LLAMA_CPP_PATH explicitly
export LLAMA_CPP_PATH=/correct/path/to/llama.cpp
make clean && make
```

### Runtime Issues

#### Error: `error while loading shared libraries: libllama.so`

```bash
# Solution 1: The build system sets rpath, but if it fails, add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLAMA_CPP_PATH/build/bin:$LD_LIBRARY_PATH
./build/benchmark_cpp --help

# Solution 2: Rebuild to ensure rpath is set correctly
make clean && make
ldd build/benchmark_cpp | grep llama  # Verify rpath

# Solution 3: Use static linking instead
# The build system automatically uses static lib if shared lib not found
```

#### Error: `Failed to load model: [model path]`

```bash
# Solution: Verify model file exists and is readable
ls -lh /path/to/model.gguf
file /path/to/model.gguf  # Should show "data" or "GGUF"

# Try with absolute path
./build/benchmark_cpp \
  -model $(pwd)/../model/embeddinggemma-300m-Q8_0.gguf \
  -mode comprehensive
```

#### Error: `Segmentation fault` or crashes

```bash
# Solution: Build debug version for detailed error messages
make clean
make debug
gdb ./build/benchmark_cpp

# In GDB:
(gdb) run -model ../model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive
# When it crashes:
(gdb) backtrace
(gdb) quit

# Common causes:
# 1. Incompatible llama.cpp version - rebuild llama.cpp from latest main
# 2. Corrupted model file - re-download model
# 3. Insufficient memory - check available RAM
```

#### Performance Issues: Much slower than expected

```bash
# Check if optimization is enabled
make info  # Should show "Build Type: Release"

# Rebuild with optimizations
make clean
make release  # Explicitly use release build

# Verify native CPU instructions are used
./build/benchmark_cpp -model ../model/embeddinggemma-300m-Q8_0.gguf \
  -mode batch -duration 5
# Should utilize AVX2/AVX512 on x86_64, NEON on ARM64

# Check CPU governor (Linux)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should be "performance", not "powersave"
sudo cpupower frequency-set -g performance  # If needed
```

### Model File Issues

#### Error: `Unsupported GGUF version`

```bash
# Solution: Update llama.cpp to latest version
cd $LLAMA_CPP_PATH
git pull origin main
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Rebuild benchmark
cd -
make clean && make
```

#### Error: `Model architecture not supported`

```bash
# This benchmark is designed for embedding models
# Verify your model is an embedding model, not a chat/completion model

# Check model metadata
$LLAMA_CPP_PATH/build/bin/llama-cli --model /path/to/model.gguf --version
```
