# Testing

## Local Pytests

```bash
pytest -s tests/test_musiccoca.py
pytest -s tests/test_gptq.py
pytest -s tests/test_prefill_correctness.py

python scripts/generate_test_reference.py
pytest -s tests/test_bitlevel_parity.py
```

## Regression test

### Benchmark tracking

```bash
python scripts/bench_track.py
python scripts/bench_show.py --samples
```

## Pull Request Template

See [pull_request_template.md](../.github/pull_request_template.md).
