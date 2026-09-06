# Changelog

## [0.1.0] — 2026-09-06

**Speculative-draft training pipeline for the Qwen family (DFLASH / DSPARK / DFLASH2), orchestrating vllm-speculators end to end.**

### Added
- **8-stage pipeline**: `regen → pretokenize → prepare → hsextract → train → convert → serve → bench`, arch-agnostic (no hardcoded model params).
- **Model presets** (`--preset`): qwen38-27b / qwen35-122b / qwen3-8b / qwen35-27b / qwen36-35b / **qwen38-flash-next**.
  - `qwen38-flash-next` carries the production `seq_len=65536` + target-layer layout `[1,7,14,20,26,32,39,45]`.
- **`--preset` seq_len normalization**: preset `seq_len` is applied when a config does not set it explicitly (`--opts seq_len=...` still wins) — removes the 8192-vs-65536 divergence between `--preset` and YAML.
- **SPECDRAFT_* environment contract**: `SPECDRAFT_SPECULATORS_REPO`, `SPECDRAFT_TRAIN_PYTHON`, `SPECDRAFT_SGLANG_PYTHON`, `SPECDRAFT_MODEL_ROOT`, `SPECDRAFT_PROXY`, `SPECDRAFT_CUDA_COMPAT` (see README Setup).
- **6 real-world integration fixes** built in: vLLM `custom_all_reduce` on Mamba-hybrid, render-free pretokenize, `BatchEncoding` normalization, exact DFLASH2 argv, multi-instance serving, sharded regen with resume.
- **Unit tests**: 9 tests covering presets / config / invoke (full-pipeline argv verification).

### Fixed
- `test_presets` now asserts the full preset set including `qwen38-flash-next` (was failing on the newly added preset).
- License unified to **Apache-2.0** (LICENSE + NOTICE), consistent with the vendored engine (vllm-speculators).
