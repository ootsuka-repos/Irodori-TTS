# Legacy pip requirements

`pyproject.toml` is the canonical dependency declaration and `uv.lock` is the
reproducible lock file. `legacy.txt` mirrors the runtime dependencies only for
tools that still require a traditional requirements file.

Prefer:

```bash
uv sync
```

For a legacy pip workflow:

```bash
python -m pip install -r requirements/legacy.txt
```
