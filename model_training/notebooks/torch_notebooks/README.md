# Torch Notebooks

Since pytorch is not very compatible with the rest of our environment, for the notebooks that need pytorch on macOS we set up a separate environemtn.

```
uv venv
source .venv/bin/activate
uv sync
jupyter notebook
```