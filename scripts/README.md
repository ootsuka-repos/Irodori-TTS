# Direct Python launchers

The installed `irodori-*` commands are the primary interface. These thin files
are kept for environments that need to invoke a Python file directly.

Run them through the project environment from the repository root, for example:

```bash
uv run python scripts/infer.py --help
uv run python scripts/train.py --help
```

| Command | Direct launcher |
| --- | --- |
| `irodori-train` | `scripts/train.py` |
| `irodori-infer` | `scripts/infer.py` |
| `irodori-prepare-manifest` | `scripts/prepare_manifest.py` |
| `irodori-convert-checkpoint` | `scripts/convert_checkpoint.py` |
| `irodori-web` | `scripts/gradio.py` |
| `irodori-voice-design-web` | `scripts/voice_design.py` |
