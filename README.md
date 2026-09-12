# specdraft

[English](#english) ｜ [中文](#中文)

<a name="english"></a>

## What is this?

An end-to-end orchestration pipeline for **training speculative-decoding draft models on the Qwen series**, covering three draft architectures:

- **DFLASH** — block-diffusion draft
- **DFLASH2** — with selector + conv heads
- **DSPARK** — Markov head + confidence head

It wraps [vllm-speculators](https://github.com/vllm-project/speculators) as the training engine and adds the glue that makes it usable for real training and serving pipelines: an 8-stage pipeline, model presets, multi-instance serving, and fixes for several real-world integration bugs.

> **Status**: v0.1.1. The 8-stage pipeline is complete; CI (GitHub Actions) validates configuration, preset resolution, and full-pipeline argv assembly via unit tests — no engine, GPU, or model is executed in CI. End-to-end training of the primary target (Qwen3.8-27B) was exercised on an internal 8×A800 node outside CI. No preset is end-to-end validated in CI; presets that are neither exercised by CI nor by the internal run are marked *untested in CI*. See [Results](#results) for independently measured performance of drafts trained with this methodology.

## Pipeline (8 stages)

```
regen        on-policy response regeneration with the target model (multi-instance sharding + resume)
  → pretokenize   chat rows → input_ids + loss_mask (bypasses the render-endpoint requirement)
  → prepare       tokenized arrow dataset (speculators prepare_data)
  → hsextract     hidden-states extraction via vLLM target-layer hooks
  → train         draft training (torchrun DP, speculators train.py)
  → convert       checkpoint → sglang-servable draft
  → serve         sglang server with speculative decoding (multi-instance, per-instance TP/CUDA masks)
  → bench         accept_length / accept_rate from sglang /metrics
```

Run all stages or any subset:

```bash
specdraft dflash  --preset qwen38-27b --stage all --dry-run   # print full argv, execute nothing
specdraft dspark  --config runs/my.yaml --stage train --dry-run
specdraft dflash2 --preset qwen38-27b --stage serve --dry-run
specdraft dspark  --config runs/my.yaml --stage all           # real run
```

## Results

`specdraft` is a training-pipeline framework; this section reports measured performance
of **drafts trained with this methodology** on independent evaluation setups. These are
not promises of specdraft throughput or of Qwen-model performance — each row is
environment-specific (see the "Evaluated on" column). Neither row measures specdraft's
own Qwen output: row 1 is a non-Qwen multimodal model, row 2 is a Qwen3.5-122B
deployment result. End-to-end training of the primary target (Qwen3.8-27B) was
exercised on an internal 8×A800 node outside CI; its production numbers are not yet
published.

| Draft architecture | Acceptance | E2E speedup vs AR | Evaluated on |
|---|---|---|---|
| DFLASH2 (trained for a multimodal OCR model) | 0.536 aggregated acceptance (online vLLM spec-count); 8.05 tokens/round | **3.08×** (C=1) / **3.02×** (C=32, capacity/min-wall), QPS throughput | single A800-80GB, vLLM 0.27.1, 100 OCR business samples, temp=0 greedy |
| DFLASH@4 (trained for Qwen3.5-122B MoE) | 2.59 tokens/round (C=32, ground-truth metrics-diff) | **1459 out tok/s** (C=32), **+59%** vs AR, same-code same-envelope | PD serving, 2×8 A800, 300 business samples (seed 42), WARMUP=1 |

Notes:
- Acceptance definition differs between rows (aggregated rate vs tokens/round); treat
  them as environment-specific, not comparable. Speedups are throughput ratios (AR=1.0),
  not latency ratios.
- The DFLASH@4 acceptance figure is the ground-truth (metrics-diff) reading; the source
  report's headline table reports a different spec-log value (3.38) for the same C=32
  config — the source report designates the metrics-diff value as the trustworthy
  acceptance measure.
- The DFLASH2/OCR row used vLLM for serving while `specdraft`'s serve stage targets
  sglang — engine-specific speedups are not directly transferable. It evidences the
  training methodology, not sglang serving performance.
- The DFLASH@4 row measured the trained draft under prefill/decode disaggregation (PD)
  on dedicated infrastructure; the number reflects deployment conditions (2×8 A800),
  not the training pipeline alone.
- An earlier internal report's figures (accept 0.486, e2e 4.8×) were superseded by the
  DFLASH2 row above and are not quoted here.
- Every figure above comes from a documented internal evaluation report; measurement
  methodology and source details are described in the v0.1.1 release notes.

## Presets

| preset | model | block/γ | target layers | notes |
|---|---|---|---|---|
| `qwen38-27b` | Qwen3.8-27B | 8/4 | 5 19 33 47 61 | ★ primary target; end-to-end trained internally on 8×A800 (not in CI) |
| `qwen35-122b` | Qwen3.5-122B-A10B | 16/7 | 1 7 14 20 26 32 39 45 | MoE; untested in CI — kept for reproducibility |
| `qwen3-8b` | Qwen3-8B | 8/4 | 2 18 33 | untested in CI; intended for smoke / single-GPU self-check |
| `qwen35-27b` | Qwen3.5-27B | 8/4 | 5 19 33 47 61 | same architecture family as 3.8-27B; untested in CI |
| `qwen36-35b` | Qwen3.6-35B-A3B | 8/4 | uniform 8 layers | MoE; untested in CI |
| `qwen38-flash-next` | Qwen3.8-Flash-Next | 8/4 | 1 7 14 20 26 32 39 45 | experimental (qwen4_exp Mamba-hybrid+MoE, seq_len 65536); untested in CI |

`model_path` resolves under `$SPECDRAFT_MODEL_ROOT`, or override it per-run (`--config` YAML / `--opts model_path=...`) with a local path or HF repo id.

## Real-world bugs this pipeline fixes

| # | problem | fix |
|---|---|---|
| 1 | vLLM `custom_all_reduce` crashes on GDN (Mamba-hybrid) models | `--disable-custom-all-reduce --distributed-executor-backend mp`; never set `VLLM_BATCH_INVARIANT=1` |
| 2 | speculators `prepare_data` requires a render-endpoint | `pretokenize` stage emits `input_ids`+`loss_mask` rows directly, so prepare skips render |
| 3 | `BatchEncoding` fails `isinstance(dict)` after transformers upgrade | normalize chat-template outputs via `hasattr`-based extraction; loss mask = suffix after the last `<\|im_start\|>` |
| 4 | DFLASH2 needs a full, exact argv (selector/conv/sliding-window flags + full vocab) | `convert`/`train` stages embed the complete argv incl. nested `dflash_config` |
| 5 | serving two 27B targets on 4 GPUs | multi-instance serve with per-instance `CUDA_VISIBLE_DEVICES` + TP assignment |
| 6 | `regen` is slow at scale | multi-instance sharded regeneration with resume |

## Setup

Requirements: Python ≥ 3.10. Training needs a venv with `torch`, `vllm`, `transformers` and `speculators`; serving needs `sglang`.

```bash
# 1) clone the engine repo (its scripts/ are invoked by this pipeline)
git clone https://github.com/vllm-project/speculators.git
#   (validated against vllm-project/speculators @ 4048017)

# 2) install this package
pip install -e .

# 3) environment
export SPECDRAFT_SPECULATORS_REPO=/path/to/speculators        # engine repo clone
export SPECDRAFT_TRAIN_PYTHON=/path/to/envs/train/bin/python  # venv with torch/vllm/transformers/speculators
export SPECDRAFT_SGLANG_PYTHON=/path/to/envs/sglang/bin/python
export SPECDRAFT_MODEL_ROOT=/path/to/local/models             # where preset model dirs live
export SPECDRAFT_PROXY=http://your-proxy:port                 # optional, for HF downloads
export SPECDRAFT_CUDA_COMPAT=/usr/local/cuda-13.0/compat      # optional, old-driver compat

# 4) sanity check
specdraft dflash --preset qwen3-8b --stage all --dry-run
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

9 unit tests cover presets / config / invoke (full-pipeline argv verification).

## End-to-end example (Qwen3.8-Flash-Next, experimental)

`--preset qwen38-flash-next` carries the target seq_len (65536) and target-layer
layout, so every stage's argv can be inspected with a dry-run. This section is an
illustration of the pipeline wiring, **not** a validated reproduction — the
flash-next preset is experimental and has not been validated end-to-end in this
repo's CI (see [Presets](#presets)):

```bash
export SPECDRAFT_SPECULATORS_REPO=/path/to/speculators
export SPECDRAFT_TRAIN_PYTHON=/path/to/envs/train/bin/python
export SPECDRAFT_SGLANG_PYTHON=/path/to/envs/sglang/bin/python
export SPECDRAFT_MODEL_ROOT=/path/to/local/models

# 8-stage dry-run (prints every argv, executes nothing)
specdraft dflash --preset qwen38-flash-next --stage all --dry-run

# stage-by-stage on real data (prep/ hs paths come from --config YAML)
#   NOTE: requires a working engine + model environment; not validated by CI.
specdraft dflash --preset qwen38-flash-next --stage train --config runs/qwen38-flash-next.yaml
specdraft dflash --preset qwen38-flash-next --stage convert
specdraft dflash --preset qwen38-flash-next --stage serve
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The engine ([speculators](https://github.com/vllm-project/speculators)) and serving stack ([sglang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm)) remain under their own licenses.

---

<a name="中文"></a>

## 中文简介

面向 **Qwen 系列投机解码草稿训练**的端到端编排框架，覆盖 DFLASH / DFLASH2 / DSPARK 三种草稿架构。以 [vllm-speculators](https://github.com/vllm-project/speculators) 为训练引擎，补齐实际训练与服务所需的 8 阶段流水线（regen → pretokenize → prepare → hsextract → train → convert → serve → bench）、模型 preset、多实例 serving，并内置 6 个真实集成坑的解法（见上方表格）。CI 仅做单元测试（argv 组装 / dry-run）；端到端训练在内部 8×A800 节点验证过（非 CI）。

环境变量：`SPECDRAFT_SPECULATORS_REPO`（引擎仓 clone）、`SPECDRAFT_TRAIN_PYTHON`（带 vllm/torch 的训练解释器）、`SPECDRAFT_SGLANG_PYTHON`（serve 解释器）、`SPECDRAFT_MODEL_ROOT`（preset 模型根目录）、`SPECDRAFT_PROXY`（可选代理）。

快速验证：

```bash
specdraft dflash --preset qwen3-8b --stage all --dry-run
pytest -q
```

License: Apache-2.0（见 [LICENSE](LICENSE)）。
