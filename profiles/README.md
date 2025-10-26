# CPU and Memory Profiles

This directory contains CPU and memory profiles from benchmark runs.

## Viewing Profiles

### CPU Profile

#### Text-based top functions:
```bash
go tool pprof -text -nodecount=20 baseline_cpu.prof
```

#### Interactive web UI:
```bash
go tool pprof -http=:8080 baseline_cpu.prof
```

#### Top application functions only (exclude runtime):
```bash
go tool pprof -text baseline_cpu.prof | grep "github.com/lth/pure-go-llamas"
```

#### Flame graph (web UI):
```bash
go tool pprof -http=:8080 baseline_cpu.prof
# Then click "VIEW > Flame Graph"
```

### Memory Profile

#### Text-based top allocations:
```bash
go tool pprof -text -nodecount=20 baseline_mem.prof
```

#### Interactive web UI:
```bash
go tool pprof -http=:8080 baseline_mem.prof
```

#### Show allocation sites:
```bash
go tool pprof -alloc_space -text baseline_mem.prof
```

## Profile Files

- `baseline_cpu.prof` - Baseline CPU performance profile (BenchmarkForward-8)
  - Generated: 2025-10-25
  - Git commit: 8c1cdf0
  - Platform: Apple M1 Pro (ARM64)
  - Duration: 3.61s
  - Total samples: 8.04s (222.82% CPU)

- `baseline_mem.prof` - Baseline memory allocation profile (BenchmarkForward-8)
  - Generated: 2025-10-25
  - Git commit: 8c1cdf0
  - Platform: Apple M1 Pro (ARM64)
  - Total allocations: 1369.44 MB

## Key Findings

### CPU Hotspots (Top 5)
1. `gguf.ParseQ8_0Block` - 3.48% (Q8_0 weight dequantization)
2. `kernels.matmulWorkerPool.processJob` - 1.87% (matrix multiplication)
3. `runtime.Model.ForwardINT8` - 3.73% (main forward pass)
4. `runtime.extractQ8_0Scales` - 3.48% (scale extraction)
5. `runtime.Model.loadLayerINT8` - 3.48% (layer loading)

### Memory Allocations (Top 5)
1. `runtime.newBufferPool` - 492 MB (35.93%)
2. `runtime.LoadModel` - 360.50 MB (26.32%)
3. `tokenizer.New` - 160.09 MB (11.69%)
4. `gguf.Reader.readMetadataValue` - 121.52 MB (8.87%)
5. `runtime.extractQ8_0Scales` - 75.41 MB (5.51%)
