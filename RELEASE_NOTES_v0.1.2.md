# Release Notes — specdraft v0.1.2

**specdraft** v0.1.2 is a real-machine validation release: the full 8-stage pipeline
(regen → pretokenize → prepare → hsextract → train → convert → serve → bench) was
executed end-to-end on A800 nodes, and every bug that only real runs could expose is
fixed.

## Verified end-to-end (outside CI)

- **Qwen3.8-27B, block 16 / γ 7 (4×A800)**: all 8 stages completed; smoke training
  (53 steps, 1 epoch on 469 regenerated samples) reached `loss_epoch=0.944`; the
  converted draft served under sglang DFLASH TP2 and benched `accept_length=1.125`
  on 50 samples. Identical run at block 8 gave `loss_epoch=1.91` / `accept_length=1.025`,
  which motivated the block-16 default.
- **Qwen3-8B (1×A800)**: full smoke — train (loss 1.604 → 0.985), convert, serve,
  bench — completed; `accept_length=1.05`.

## Fixed: 8 bugs found only by real runs

Cross-stage data-flow (regen merge clobbering pretokenize output), engine CLI
contract drift (`--max-samples 0` semantics, flag renames, duplicate flags resolved
only by engine last-wins), missing `--model` on regen (all requests 404'd),
convert checkpoint-path mismatch, flashinfer version assertion between vLLM and
sglang, and an `expandable_segments` incompatibility during hidden-state extraction.
Details in CHANGELOG.md and the repo's validation report.

## Changed

- `qwen38-27b` preset defaults to **block 16 / γ 7** (reference DFlash block size).
- README: Status and preset table reflect the two verified presets; bugs table row 7
  (cross-stage data flow + engine CLI contract); Setup documents the flashinfer
  conflict and the `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` mitigation.

## Notes

- Other presets (`qwen35-27b`, `qwen36-35b`, `qwen38-flash-next`) remain *untested in
  CI*: their smoke runs are blocked on environment/dependency-matrix issues that are
  documented in the validation report, not on pipeline bugs.
- Engine pin unchanged (`vllm-project/speculators @ 4048017`).
