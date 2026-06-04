# Installation

Magenta RealTime 2 ships as a single PyPI package, `magenta-rt`, which gives you
both the `mrt` command-line tool and the `magenta_rt` Python library. Which
backend you install depends on what you want to do:

- **Real-time streaming** (DAW plugins, live performance) runs on **Apple
  Silicon** via the **MLX** backend. Install `magenta-rt[mlx]`.
- **Offline / batch generation and research** runs anywhere via the **JAX**
  backend, which is always included. The base install ships JAX on CPU; on Linux
  you add a hardware-accelerated JAX wheel (CUDA or TPU).

> The base install always includes JAX (CPU). The `[mlx]` extra adds the Apple
> Silicon backend on top of it — it does not replace JAX.

For which Macs can stream each model size in real-time, see the
[hardware requirements table](models.md#hardware-requirements).

## 1. Create a virtual environment

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a Python 3.12 virtual environment
uv venv --python 3.12
source .venv/bin/activate
```

## 2. Install `magenta-rt`

::::{tab-set}

:::{tab-item} macOS (Apple Silicon)
```bash
uv pip install "magenta-rt[mlx]"
```
:::

:::{tab-item} Linux (NVIDIA / TPU)
Pick the [JAX wheel](https://docs.jax.dev/en/latest/installation.html) that
matches your hardware (e.g. `jax[cuda13]` or `jax[tpu]`) and install it alongside
`magenta-rt`:

```bash
uv pip install "magenta-rt" "jax[cuda13]"
```
:::

::::

## 3. Download models

```bash
# Download shared resources (MusicCoCa style model + SpectroStream codec)
mrt models init

# Download a streaming model — run with no argument to pick interactively
mrt models download
```

Assets are saved under `~/Documents/Magenta/magenta-rt-v2/`. See
[Models & checkpoints](models.md) for the full directory layout, the available
language-model checkpoints, and how to fetch raw safetensors for research.

## 4. Generate music

Confirm everything works by generating a short clip. Use `mrt mlx` on Apple
Silicon and `mrt jax` elsewhere:

::::{tab-set}

:::{tab-item} macOS (MLX)
```bash
# Use --model=mrt2_small for the small model
mrt mlx generate --prompt "disco funk" --duration 4.0 --model=mrt2_base
```
:::

:::{tab-item} Linux (JAX)
```bash
mrt jax generate --prompt "disco funk" --duration 4.0 --model=mrt2_base
```
:::

::::

See [Inference](inference.md) for more on prompting, tokens, and bulk generation.

## Local development

To work on the library itself, clone the repo and install in editable mode
instead of from PyPI:

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cd magenta-realtime

uv pip install -e ".[mlx]"           # macOS
uv pip install -e "." "jax[cuda13]"  # Linux with CUDA
```

## C++ app development

To build C++ apps on the inference engine, install cmake and build a target.
This assumes you have already downloaded models (step 3 above).

```bash
# Install cmake
uv pip install "cmake<3.28"

# Build hello_mrt2, a basic command-line interface
cmake . -B build
cmake --build build --target hello_mrt2 -j10

# Generate 4 seconds of music (replace <model_name>, e.g. mrt2_small)
./build/examples/hello_mrt2/hello_mrt2 \
    ~/Documents/Magenta/magenta-rt-v2/models/<model_name>/<model_name>.mlxfn \
    ~/Documents/Magenta/magenta-rt-v2/resources \
    100 \
    --prompt "ambient pads with sub bass"
```
