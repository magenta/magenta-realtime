# `magenta_rt.mlx`

Authoritative pure-MLX inference implementation of Magenta-RT built on top of the **`sequence_layers.mlx`** framework.

This subpackage represents the original production codebase target, modeling modules as explicit stateful sequence combinators (`sl.SequenceLayer`, `sl.Emitting`) threaded with runtime latency tracking buffers.

## Quick start

The fastest route to generate audio using a safetensors checkpoint from the shell:

```sh
mrt mlx generate \
    --model mrt2_small \
    --checkpoint mrt2_small.safetensors \
    --duration 4.0
```

For programmatic model building and weight loading directly in your own application code:

```python
import mlx.core as mx
import sequence_layers.mlx as sl
from magenta_rt.mlx import (
    model as mlx_model, spectrostream as mlx_ss, system as mlx_system,
)
from magenta_rt.mlx.load_weights import load_weights

# 1. Resolve the target model specification
spec = mlx_model.get_model_class("mrt2_small")()

# 2. Assemble the full SequenceLayer combinator system
mrt = mlx_system.MagentaRT2Sampler.Config(
    depthformer=spec.depthformer_config(),
    spectrostream=mlx_ss.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=spec.spectrostream.rvq_truncation_level,
        use_unique_codes=False,
    ),
    int16_outputs=False,
).make()

# 3. Load parameters directly from disk into the initialized system
load_weights(
    mrt, "checkpoints/mrt2_small.safetensors",
    num_input_channels=spec.input_num_channels,
)

# 4. Materialize deferred state allocations (eager compilation initialization)
input_spec = sl.ChannelSpec(shape=(spec.input_num_channels,), dtype=mx.int32)
from sequence_layers.mlx import export as sl_export
sl_export._materialize_deferred(
    mrt, batch_size=1, input_spec=input_spec,
    constants={
        "classifier_free_guidance_scale_musiccoca": mx.array([1.0]),
        "classifier_free_guidance_scale_notes": mx.array([1.0]),
        "temperature": mx.array([1.0]),
        "top_k": mx.array([40], dtype=mx.int32),
    },
)
```

## Public API Surface

| Module | Primary Exports |
|--------|-----------------|
| `mlx.model` | `ModelSpec`, `TokensConfig`, full model spec registry (`MODEL_REGISTRY`, `get_model_class`), configuration building wrappers |
| `mlx.system` | Top-level orchestrator combinator (`MagentaRT2Sampler`, `MagentaRT2Sampler.Config`), `convert_from_unique_codes` |
| `mlx.spectrostream` | Codec subpackage hosting `SpectroStream`, `SpectroStreamConfig`, and the standard preset (`stft_spectrostream_40ms_generic_48khz_stereo_config`) |
| `mlx.depthformer` | Core network definitions (`MultivariateDecoder`, `EncoderDecoder`, `StreamingEncoderDecoderSampler`) built as custom `sl.Emitting` layers |
| `mlx.transformer` | Base network combinator structures (`SLTransformer`, `SLTransformerFFN`, `MultiChannelEmbedding`) |
| `mlx.load_weights` | Main deserializer loader (`load_weights`), pure parameter setting tools (`_set_param`, `_collect_all_params`) |
| `mlx.generate` | Standalone execution application driver loop |

## Running Tests

Regression and parity checks reside under the primary integration root:
```sh
pytest tests/mlx -m "not slow and not checkpoint"
```
