# Contributing

Thanks for considering contributing to **specdraft**.

## Development setup

```bash
git clone <your-fork>/specdraft
cd specdraft
pip install -e ".[dev]"

# engine + model env (needed to run the pipeline; unit tests only need the paths)
export SPECDRAFT_SPECULATORS_REPO=/path/to/speculators
export SPECDRAFT_TRAIN_PYTHON=/path/to/envs/train/bin/python
export SPECDRAFT_SGLANG_PYTHON=/path/to/envs/sglang/bin/python
export SPECDRAFT_MODEL_ROOT=/path/to/local/models
```

## Tests

```bash
pytest -q     # 85 unit tests: presets / config / invoke / error paths
```

Run with `SPECDRAFT_TRAIN_PYTHON`/`SPECDRAFT_SPECULATORS_REPO` set — the unit tests assemble argv but do not execute the engine, so a plain python (or any python on PATH) works as `SPECDRAFT_TRAIN_PYTHON`.

## Conventions

- **Python ≥ 3.10**, no heavy deps beyond `pydantic` / `PyYAML` / `safetensors`.
- New model support = add a `QwenPreset` entry (`specdraft/presets.py`) **and** a `presets/<name>.yaml`; keep `seq_len` consistent between the two (the target seq_len lives on the preset).
- `--preset` vs `--config` semantics: config YAML overrides preset fields; `--opts k=v` overrides both.
- Every PR: update `CHANGELOG.md`, add/extend a unit test in `tests/`, keep `pytest -q` green.
- New scripts under `vendor/` must carry the Apache-2.0 header (see existing files).

## Commit messages

Conventional commits: `feat(framework): ...`, `fix(models): ...`, `perf(hs-prefetch): ...`, `docs: ...`.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). By contributing you agree to license your contribution under the same terms.
