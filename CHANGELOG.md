# Changelog

## [0.1.1] — 2026-09-12

**Documentation-and-validation polish release: honesty corrections + a sourced Results section.**

### Changed
- **Validation status made explicit**: README Status now separates "what CI validates"
  (unit tests: config / preset resolution / full-pipeline argv assembly — no engine,
  GPU, or model runs) from "what was exercised internally" (end-to-end Qwen3.8-27B
  training on an 8×A800 node, outside CI).
- **Preset table**: `qwen35-122b` / `qwen35-27b` / `qwen36-35b` / `qwen38-flash-next`
  marked *untested in CI*; `qwen3-8b` marked *not run by CI*; `qwen38-27b` marked
  *primary target, trained internally (not in CI)*.
- **`qwen38-flash-next` downgraded to experimental** (qwen4_exp Mamba-hybrid+MoE):
  "production" wording removed from README, CHANGELOG, preset YAML, `presets.py`,
  `config.py` comment, and a test comment.
- **New README "Results" section**: measured performance of drafts trained with this
  methodology (multimodal OCR; Qwen3.5-122B MoE under PD serving), each row with its
  acceptance metric, speedup definition, evaluation environment, and caveats. Source
  file paths per figure are listed in `SPECSDRAFT_POLISH_REPORT.md`.
- **Test suite hardened (9 → 85 tests)**: per-preset argv smoke for all 6 presets ×
  3 draft types on both config paths (`--preset` table cross-checked against
  `presets/*.yaml`), full-pipeline dry-run verification per preset, and
  config-resolution error paths (readable `error:` messages on stderr, no bare
  tracebacks). Making the smoke tests honest surfaced and fixed real defects: the
  README sanity-check command crashed on missing data paths (dry-run now prints an
  explicit skip note), `cmd_train` crashed on unset data paths in dry-run (now
  `<unset>` placeholder), user errors printed bare tracebacks (now one readable line),
  `--opts seq_len=abc` silently injected a string into the engine argv (now a pydantic
  ValidationError), the `qwen36-35b` YAML target layers diverged from the python preset
  table (`[1,6,11,16,22,27,32,37]` → `[1,6,12,17,22,27,33,38]`), a corrupted `train.*`
  opts override could emit a contradictory duplicate flag in the train argv (override
  now replaces), `--stage ','` silently meant "all stages" (now rejected), and
  `--stage` gained comma-separated subset support matching the README examples.
- **Version bumped to 0.1.1** (`pyproject.toml` + `specdraft/__init__.py`).

## [0.1.0] — 2026-09-06

**Speculative-draft training pipeline for the Qwen family (DFLASH / DSPARK / DFLASH2), orchestrating vllm-speculators end to end.**

### Added
- **8-stage pipeline**: `regen → pretokenize → prepare → hsextract → train → convert → serve → bench`, arch-agnostic (no hardcoded model params).
- **Model presets** (`--preset`): qwen38-27b / qwen35-122b / qwen3-8b / qwen35-27b / qwen36-35b / **qwen38-flash-next**.
  - `qwen38-flash-next` carries the target `seq_len=65536` + target-layer layout `[1,7,14,20,26,32,39,45]` (experimental preset, not validated end-to-end in CI).
- **`--preset` seq_len normalization**: preset `seq_len` is applied when a config does not set it explicitly (`--opts seq_len=...` still wins) — removes the 8192-vs-65536 divergence between `--preset` and YAML.
- **SPECDRAFT_* environment contract**: `SPECDRAFT_SPECULATORS_REPO`, `SPECDRAFT_TRAIN_PYTHON`, `SPECDRAFT_SGLANG_PYTHON`, `SPECDRAFT_MODEL_ROOT`, `SPECDRAFT_PROXY`, `SPECDRAFT_CUDA_COMPAT` (see README Setup).
- **6 real-world integration fixes** built in: vLLM `custom_all_reduce` on Mamba-hybrid, render-free pretokenize, `BatchEncoding` normalization, exact DFLASH2 argv, multi-instance serving, sharded regen with resume.
- **Unit tests**: 9 tests covering presets / config / invoke (full-pipeline argv verification).

### Fixed
- `test_presets` now asserts the full preset set including `qwen38-flash-next` (was failing on the newly added preset).
- License unified to **Apache-2.0** (LICENSE + NOTICE), consistent with the vendored engine (vllm-speculators).
