# Magenta RealTime 2 documentation

This directory holds the Sphinx source for the Magenta RealTime 2
documentation site, published to GitHub Pages by
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on every push to
`main` that touches `docs/`.

## Build locally

```bash
uv pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The pages are authored in Markdown (MyST). `conf.py` configures the
`sphinx-book-theme` and the MyST extensions; `index.md` defines the navigation
via `toctree` directives.
