# Release Notes — specdraft v0.1.4

**v0.1.4 is a consolidated release.** Everything that accumulated after v0.1.3 —
two unmerged fix branches, the ⑥ specdraft patch from the training audit, and a
repository-wide sweep for loose ends — ships as **one version**. If you are
upgrading from any v0.1.x, this is the only release in between that matters.

Contents merged here:

- `docs/housekeeping-0920` — dangling-reference fixes, README status, verification-bundle link
- `fix/convert-sample-from-anchor` — convert layout self-description + null-default bug
- ⑥ patch (specdraft audit, 2026-09-21) — convert `sample_from_anchor` fix (fully
  subsumed by the branch above; zero delta) + the qwen38-flash-next preset
  corrections (applied here)
- Full-repo audit — version-string alignment, link check, test-count drift

## What's new

- **Servable configs are self-describing about layout.** `convert` now writes
  `sample_from_anchor` into the dflash2 nested `dflash_config` (and resolves the
  dspark field's default at export), so a converted draft declares the layout it
  was trained with instead of inheriting whatever the serving engine's default
  happens to be. This is the fix for the "checkpoint silently served shifted by
  one position" class of failure.
- **`qwen38-flash-next` preset now mirrors the real dflash2 training run.** Field
  values were read verbatim from the `dflash2_q4_b4h24` train command:
  block 4 / γ 4, seq_len 16384, epochs 5, nproc 4. The preset header documents
  the dflash2 selector/conv architecture values (emitted automatically by the
  pipeline for `draft=dflash2`) and discloses the three switches the pipeline
  does not emit by default (`--optimizer muon`, `--noise-std 0.05`,
  `--muon-ns-steps 3`) with the exact `--opts train.*` passthrough to supply them.
- **README "Independent verification" section** linking the
  `verification/sglang-38191-122b-mtp3-pd-soak` bundle: a 1-hour
  prefill/decode-disaggregation soak of Qwen3.5-122B MTP3 on sglang
  (8×A800 decode node, Mooncake RDMA), 46,826/46,826 requests OK, tracked as
  [sglang#38191](https://github.com/sgl-project/sglang/issues/38191).
- `docs/RESULTS_SOURCES.md` — sanitized figure provenance for the README Results
  section (replaces the internal `SPECSDRAFT_POLISH_REPORT.md` references).

## Fixes

- **convert null-default bug (dspark + dflash2)**: an absent
  `sample_from_anchor` was exported as JSON `null`; the engine evaluates
  `bool(cfg.get("sample_from_anchor", True))`, so `null` → `False` → the
  checkpoint was silently served as 1+N although the speculators default is
  True (anchor-first). Export now resolves the default: `bool(src.get(..., True))`.
- **qwen38-flash-next preset values**: `block_size` 8 → 4, `seq_len` 65536 →
  16384, `epochs` 10 → 5, `nproc` 8 → 4. (65536 was the backbone-training
  sequence length; the draft itself trained at 16384.) `target_layer_ids`
  `[1,7,14,20,26,32,39,45]` unchanged, as ruled.
- Dangling document references to the internal polish report repointed to
  `docs/RESULTS_SOURCES.md` (RELEASE_NOTES_v0.1.1 ×2, CHANGELOG v0.1.1 ×1).
- README unit-test count 85 → 86 (actual since v0.1.3).

## Known limitations

- **Aux-alignment retraining conclusion is pending** — the marathon run is still
  in progress; no acceptance verdict on the retrained alignment yet.
- **qwen4 checkpoint awaits the OCR training line update** — the qwen4-side
  checkpoint refresh is gated on that line, so qwen38-flash-next remains
  *experimental / untested in CI* even with corrected preset values.
- The specdraft pipeline does not emit `--optimizer muon`, `--noise-std 0.05`,
  or `--muon-ns-steps 3` by default (the real dflash2 run passed them
  explicitly). Supply them via engine passthrough
  (`--opts train.optimizer=muon train.noise_std=0.05 train.muon_ns_steps=3`)
  when reproducing that run exactly. Whether `RunConfig.optimizer` should be
  wired into the train argv is left for a follow-up decision.
- Acceptance figures from short smoke runs sit near the 1.0 baseline; they
  certify pipeline mechanics, not draft quality.

## Compatibility

- Engine pin unchanged: validated against `vllm-project/speculators @ 4048017`.
  The hs-prefetch options remain downstream-only (see README "Engine scope").
- No CLI, schema, or preset-name changes: existing run YAMLs keep working. The
  only behavioral change is the corrected qwen38-flash-next default block size
  and sequence length, and convert output now always carries an explicit
  `sample_from_anchor` (a no-op for checkpoints that already declared it).
- 86 unit tests green on Python 3.10–3.12 (CI matrix), no engine/GPU needed.

---

*Previous release notes (v0.1.1–v0.1.3) are retained in this repo for history;
CHANGELOG.md treats v0.1.4 as the single consolidation point for the v0.1.4
work stream.*
