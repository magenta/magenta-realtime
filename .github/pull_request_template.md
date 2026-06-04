# Title

Description.

## Related Issues

* Tag any related issues here.

## Local Pytests

> [!NOTE]
> Use `mrt models init` to download the necessary resources. Then use `mrt checkpoints download` to download `mrt2_small.safetensors` and `mrt2_base.safetensors`

I ran
```bash
mrt jax generate --model=mrt2_small
mrt mlx generate --model=mrt2_small
mrt mlx generate --model=mrt2_small --no-mlxfn --bits=8

pytest -s tests/test_musiccoca.py
pytest -s tests/test_prefill_correctness.py

python scripts/generate_test_reference.py
pytest -s tests/test_bitlevel_parity.py
```
and observed the following output:
```

```

## Benchmark Regression Test

I ran
```bash
python scripts/bench_track.py
python scripts/bench_show.py --samples
```
and observed the following output:
```

```
