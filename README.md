# specdraft

[English](#english) ｜ [中文](#中文)

<a name="english"></a>

## What is this?

An end-to-end orchestration pipeline for **training speculative-decoding draft models on the Qwen series**, covering three draft architectures:

- **DFLASH** — block-diffusion draft
- **DFLASH2** — with selector + conv heads
- **DSPARK** — Markov head + confidence head

It wraps [vllm-speculators](https://github.com/vllm-project/speculators) as the training engine and adds the glue that makes it production-usable: an 8-stage pipeline, model presets, multi-instance serving, and fixes for several real-world integration bugs.

> **Status**: v0.1.0. The pipeline is complete and unit-tested; end-to-end runs are verified on an 8×A800 node. Large-model presets (122B/35B MoE) are provided for reproducibility but marked *untested in CI*.

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

## Presets

| preset | model | block/γ | target layers | notes |
|---|---|---|---|---|
| `qwen38-27b` | Qwen3.8-27B | 8/4 | 5 19 33 47 61 | ★ primary; aligned with the reference DFLASH2 draft config |
| `qwen35-122b` | Qwen3.5-122B-A10B | 16/7 | 1 7 14 20 26 32 39 45 | MoE; verify cost makes draft training often unprofitable — kept for reproducibility |
| `qwen3-8b` | Qwen3-8B | 8/4 | 2 18 33 | smoke test / single-GPU self-check |
| `qwen35-27b` | Qwen3.5-27B | 8/4 | 5 19 33 47 61 | same architecture family as 3.8-27B |
| `qwen36-35b` | Qwen3.6-35B-A3B | 8/4 | uniform 8 layers | MoE |
| `qwen38-flash-next` | Qwen3.8-Flash-Next | 8/4 | 1 7 14 20 26 32 39 45 | ★ qwen4_exp Mamba-hybrid+MoE; seq_len 65536 |

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

## End-to-end example (Qwen3.8-Flash-Next)

`--preset qwen38-flash-next` carries the production seq_len (65536) and target-layer
layout, so the full pipeline is reproducible from the preset alone:

```bash
export SPECDRAFT_SPECULATORS_REPO=/path/to/speculators
export SPECDRAFT_TRAIN_PYTHON=/path/to/envs/train/bin/python
export SPECDRAFT_SGLANG_PYTHON=/path/to/envs/sglang/bin/python
export SPECDRAFT_MODEL_ROOT=/path/to/local/models

# 8-stage dry-run (prints every argv, executes nothing)
specdraft dflash --preset qwen38-flash-next --stage all --dry-run

# stage-by-stage on real data (prep/ hs paths come from --config YAML)
specdraft dflash --preset qwen38-flash-next --stage train --config runs/qwen38-flash-next.yaml
specdraft dflash --preset qwen38-flash-next --stage convert
specdraft dflash --preset qwen38-flash-next --stage serve
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The engine ([speculators](https://github.com/vllm-project/speculators)) and serving stack ([sglang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm)) remain under their own licenses.

---

<a name="中文"></a>

## 中文简介

面向 **Qwen 系列投机解码草稿训练**的端到端编排框架，覆盖 DFLASH / DFLASH2 / DSPARK 三种草稿架构。以 [vllm-speculators](https://github.com/vllm-project/speculators) 为训练引擎，补齐生产化所需的 8 阶段流水线（regen → pretokenize → prepare → hsextract → train → convert → serve → bench）、模型 preset、多实例 serving，并内置 6 个真实集成坑的解法（见上方表格）。

环境变量：`SPECDRAFT_SPECULATORS_REPO`（引擎仓 clone）、`SPECDRAFT_TRAIN_PYTHON`（带 vllm/torch 的训练解释器）、`SPECDRAFT_SGLANG_PYTHON`（serve 解释器）、`SPECDRAFT_MODEL_ROOT`（preset 模型根目录）、`SPECDRAFT_PROXY`（可选代理）。

快速验证：

```bash
specdraft dflash --preset qwen3-8b --stage all --dry-run
pytest -q
```

License: Apache-2.0（见 [LICENSE](LICENSE)）。
