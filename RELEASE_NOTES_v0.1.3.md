# Release Notes — specdraft v0.1.3

specdraft v0.1.3 is a single-fix hotfix on top of v0.1.2. No new stages, presets,
or engine features; the preset table and default block sizes are unchanged.

## Fixed

- **hsextract: single-GPU KV-cache OOM** — hidden-states extraction on one GPU
  could request a KV cache larger than the memory left after verifier weights
  load (Qwen3.5-27B: 8.4 GiB needed vs ~7 GiB free), crashing vLLM at startup
  before any extraction ran. TP=1 now caps `max-model-len` to a KV-friendly
  value; `extract_max_len` overrides.

## Validation

- Exercised on a real A800 node: single-GPU extraction now completes on
  Qwen3.5-27B and Qwen3.6-35B-A3B (both crashed at vLLM startup before the
  fix); the multi-GPU / TP2 path is unchanged.
- 86 unit tests green (config / preset resolution / per-preset argv assembly,
  no engine or GPU in CI).

## Known limitations (unchanged from v0.1.2)

- Acceptance figures from short smoke runs sit near the 1.0 baseline; they
  certify pipeline mechanics, not draft quality.
