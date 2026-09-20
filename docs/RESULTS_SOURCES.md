# Results sources & honesty audit

This document backs the figures quoted in the README [Results](../README.md#results)
section and the v0.1.1 release notes. It replaces an earlier internal working report:
the internal file names and paths are deliberately not reproduced here — what matters
for reproducibility is *which evaluation each figure came from, under which metric
definition, and with which caveat*. Internal evaluations are identified below by
neutral labels (E1, E2).

## Figure provenance

### Row 1 — DFLASH2 (multimodal OCR draft)

| field | value |
|---|---|
| source evaluation | **E1**: a documented internal evaluation of a DFLASH2 draft trained with this methodology for a multimodal OCR target |
| serving engine | vLLM 0.27.1, single A800-80GB |
| workload | 100 OCR business samples, `temp=0` (greedy) |
| acceptance metric | aggregated acceptance as reported by vLLM's spec-count (online counter) |
| speedup metric | throughput ratio vs the same model served autoregressively (AR = 1.0); reported at C=1 and C=32 (capacity / min-wall settings) |
| caveats | engine-specific: this row used vLLM serving while `specdraft`'s serve stage targets sglang — it evidences the training methodology, not sglang serving performance |

### Row 2 — DFLASH@4 (Qwen3.5-122B MoE draft)

| field | value |
|---|---|
| source evaluation | **E2**: a documented internal evaluation of a DFLASH@4 draft trained with this methodology for Qwen3.5-122B, served under prefill/decode disaggregation |
| serving engine | sglang, PD serving with a 2×8 A800 deployment (dedicated infrastructure) |
| workload | 300 business samples (seed 42), WARMUP=1, C=32 |
| acceptance metric | ground-truth tokens/round from a metrics-diff over the serving logs |
| speedup metric | output-token throughput (1459 out tok/s at C=32), +59% vs AR, same-code same-envelope comparison |
| caveats | reflects deployment conditions (2×8 A800 under PD), not the training pipeline alone |

### Known divergence inside E2 (disclosed in README)

The source report's headline table also quotes a spec-log acceptance value of **3.38**
for the same C=32 config. The source report designates the metrics-diff value
(**2.59**) as the trustworthy acceptance measure; README quotes only the latter. The
spec-log number is retained here only as a record of the divergence.

## Honesty audit (v0.1.1)

- An earlier internal report quoted **accept 0.486 / e2e 4.8×** for the same OCR
  target family. Those figures were superseded by the E1 measurement methodology
  and are **not quoted** anywhere in this repository.
- The README Results table does not aggregate figures across evaluations: each row
  carries its own environment column and the table notes state that the acceptance
  definitions differ between rows and are not comparable.
- Every preset CI does not exercise end-to-end is marked *untested in CI* in the
  preset table; `qwen38-flash-next` is marked *experimental*.
- CI scope is stated explicitly: unit tests validate configuration, preset
  resolution, and argv assembly only — no engine, GPU, or model runs in CI.

## Hardening backlog (not yet taken)

- **Engine pin refresh**: README validates against `vllm-project/speculators @
  4048017`; no newer engine version has been re-validated end-to-end.
- **`qwen38-flash-next` end-to-end validation**: the preset is wired and
  dry-run-testable, but has not been validated on a real run in this repo.
- **Spec-log vs metrics-diff reconciliation**: E2's two acceptance readings have not
  been root-caused to a single reconciled counter; until then, README quotes the
  metrics-diff reading only.
- **Qwen3.8-27B (block 16) results publication**: the primary target has been
  exercised end-to-end internally (regen → bench); its production numbers are not
  yet published pending a longer, reportable training run.
