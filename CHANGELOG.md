# Changelog

## [0.1.3] — 2026-09-15

**Hotfix: single-GPU hidden-states extraction no longer OOMs at vLLM startup.**

### Fixed
- `hsextract` on a single GPU could request a KV cache larger than the memory left
  after verifier weights load (Qwen3.5-27B: 8.4 GiB needed vs ~7 GiB free), crashing
  vLLM before extraction started. TP=1 now caps `max-model-len` to a KV-friendly
  value; `extract_max_len` overrides.

### Validation
- Exercised on a real A800 node: single-GPU extraction on Qwen3.5-27B and
  Qwen3.6-35B-A3B (previously crashing), 4×A800 TP2 path unchanged.
- 86 unit tests green.

## [0.1.2] — 2026-09-13

**Real-machine validation release: the 8-stage pipeline verified end-to-end on A800s.**

### Fixed (8 bugs found only by running the pipeline on real hardware/stages)
- `--stage all --dry-run` crashed on the first data-dependent stage; dry-run now
  prints `(skipped: requires ...)` and continues.
- `regen` merge output overwrote the pretokenize input; `pretokenize` then truncated
  merged data to 0 bytes — stage outputs now use dedicated paths.
- `cmd_regen` did not pass `--model`, so the vLLM server registered the model under
  its full path and every request 404'd.
- `--max-samples 0` (newer engine semantic: "process nothing") produced an empty
  hidden-states set; the flag is only forwarded when explicitly set.
- Engine flag renames (`max_steps` → `--max-steps`) are translated; framework-owned
  flags passed through `--opts` replace the built-in flag instead of duplicating it.
- `convert` looked for `ckpt/config.json` while the engine saves under a checkpoint
  subdirectory.
- The serve stage now sets `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` to tolerate the
  flashinfer version pinned by vLLM 0.27.x.
- `expandable_segments` is not set during hidden-states extraction (incompatible
  with the KV connector path).

### Changed
- **`qwen38-27b` preset defaults to block 16 / γ 7** (reference DFlash block size).
  On an identical 53-step smoke run, block 16 trained to lower loss (0.944 vs 1.91)
  and served with higher acceptance (1.125 vs 1.025) than block 8.
- README Status/preset table: `qwen38-27b` and `qwen3-8b` marked as verified
  end-to-end outside CI; bugs table gained a 7th row (cross-stage data flow +
  engine CLI contract); Setup documents the flashinfer version conflict.

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
  details per figure are documented in [docs/RESULTS_SOURCES.md](docs/RESULTS_SOURCES.md).
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
