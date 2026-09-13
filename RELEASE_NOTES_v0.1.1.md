# Release Notes — specdraft v0.1.1

**specdraft** v0.1.1 is a documentation-and-validation polish release on top of v0.1.0.
No new pipeline stages or engine features were added; the focus is honesty and
verifiability for open-source readers.

## What changed

### Validation status made explicit (README Status + Presets)

- The Status note now distinguishes **what CI validates** (configuration, preset
  resolution, and full-pipeline argv assembly via unit tests — no engine, GPU, or
  model is executed in CI) from **what was exercised internally** (end-to-end
  training of the primary Qwen3.8-27B target on an 8×A800 node, outside CI).
- The preset table now marks every preset that CI does not exercise end-to-end:
  - `qwen35-122b`, `qwen35-27b`, `qwen36-35b`, `qwen38-flash-next` → *untested in CI*
  - `qwen3-8b` → *untested in CI; intended for smoke / single-GPU self-check*
  - `qwen38-27b` → *primary target; end-to-end trained internally (not in CI)*

### `qwen38-flash-next` downgraded to experimental

- The Flash-Next preset (qwen4_exp Mamba-hybrid + MoE, seq_len 65536) was previously
  described as a *production target*. It is not validated end-to-end in this repo's
  CI, so all references were downgraded to **experimental / untested in CI**:
  README preset table + Status + end-to-end example section, CHANGELOG, the preset
  YAML header, `presets.py` notes, a `config.py` comment, and a test comment.

### New "Results" section (README)

- Added a measured-results table for drafts **trained with this methodology** on
  independent evaluation setups (multimodal OCR, and Qwen3.5-122B MoE under PD
  serving). Each row states the acceptance metric, the speedup definition
  (throughput ratio vs AR), the exact evaluation environment, and the metric caveats.
- These figures are environment-specific measurements, not specdraft performance
  promises; source-file paths for every figure are listed in
  `SPECSDRAFT_POLISH_REPORT.md`.
- An earlier internal report's figures (accept 0.486, e2e 4.8×) were superseded and
  are not quoted.

### CLI honesty and robustness fixes (with tests 9 → 85)

Expanding the test suite surfaced ten real bugs, all fixed in this release:

- `--stage all --dry-run` (the README sanity-check command) crashed on the first
  data-dependent stage; dry-run now prints an explicit `(skipped: requires ...)`
  and continues, so users can inspect every stage's argv.
- The `--preset` table path and the `--config` YAML path had drifted apart
  (different `seq_len` for 5/6 presets, a wrong layer list in `qwen36-35b.yaml`);
  both paths now resolve identically, with a cross-check test that fails if the
  table and the YAML ever diverge again.
- `train.epochs=N` passthrough emitted a duplicate `--epochs` flag that only
  worked by engine last-wins accident; overrides now replace the hardcoded flag.
- All user errors (unknown preset/stage, malformed YAML, missing files or env
  vars, bad `--opts` values) now exit 1 with a readable `error: ...` message
  instead of a bare traceback; `--opts` values are type-validated.
- Tests grew from 9 to 85: per-preset argv smoke for all 6 presets × 3 draft
  types across both config paths, full-pipeline dry-run verification, and 17
  config-resolution error-path tests.

### Engine pin unchanged

- README still states validation against `vllm-project/speculators @ 4048017`. The
  pin was **not** bumped: it reflects the version actually exercised, and no newer
  engine version was validated as part of this release.

## Changelog

- README: honesty pass on Status / preset table / flash-next example; new Results
  section with sourced figures.
- CHANGELOG: flash-next marked experimental.
- presets/qwen38-flash-next.yaml, specdraft/presets.py, specdraft/config.py,
  tests/test_core.py: "production" → "experimental / target" wording.
- Version bumped to **0.1.1** (pyproject.toml + `specdraft/__init__.py`).

## Notes

- Full source-file paths behind every Results figure, the honesty-audit findings, and
  an assessment of the not-yet-taken hardening work are in `SPECSDRAFT_POLISH_REPORT.md`
  (shipped in the same branch).
- Behavior changes are limited to the CLI-honesty fixes listed above; the pipeline
  stages and engine integration are unchanged. All 85 tests pass.
