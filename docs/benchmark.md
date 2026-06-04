### Latency benchmark

Prerequisites:

```bash
uv pip install "cmake<3.28"
```

Install [Xcode](https://developer.apple.com/xcode/) for extra dependencies related to `metal` to be installed.

Run the benchmark:
```bash
cmake core/src/benchmark -B benchmark_build
cmake --build benchmark_build --target benchmark_mlxfn -j10

# Benchmark latency:
./benchmark_build/benchmark_mlxfn ~/Documents/Magenta/magenta-rt-v2/models/mrt2_base
```
