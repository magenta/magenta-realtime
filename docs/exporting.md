# Exporting

## Depthformer: safetensors -> mlxfn

We export the depthformer model to mlxfn for the C++ inference engine.

```bash
# Standard export:
mrt mlx export --output-name=mrt2_base --bits=8

# Export with GPTQ if int4 is needed
mrt mlx export --output-name=mrt2_base --bits=4 --quantize-method=gptq --gptq-cal-steps=128

# Export a custom untrained model spec:
mrt mlx export --output-name=debug --num-layers=2 --depth-num-layers=2 --bits=8 --skip-restore
```

This exports `.mlxfn` and `.safetensors` fresh model state into
`~/Documents/Magenta/magenta-rt-v2/models/<name>`.

## SpectroStream encoder: safetensors -> mlxfn

The SpectroStream encoder is used for audio prefilling in the plugin. Export the `.mlxfn` file:

```bash
mrt mlx export-spectrostream
```

By default, this exports to `~/Documents/Magenta/magenta-rt-v2/resources/spectrostream/spectrostream_encoder.mlxfn`.
