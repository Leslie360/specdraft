"""Qwen preset table + derive helpers.

Each preset provides: model path / layers / hidden / vocab / arch / block / gamma / target_layer_ids.
target_layer_ids can be auto-derived (uniform N layers) or explicit (e.g. the reference DFLASH2
draft's [5,19,33,47,61]).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Model root: every preset's model_path resolves under $SPECDRAFT_MODEL_ROOT
# (or override per-run via config/yaml model_path with an HF repo id or a local path)
MODEL_ROOT = os.environ.get("SPECDRAFT_MODEL_ROOT", ".")


@dataclass(frozen=True)
class QwenPreset:
    name: str
    model_path: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    arch: str            # qwen3 / qwen3_5_text / qwen3_5_moe_text ...
    block_size: int
    decay_gamma: int     # matches block (paper: b16→7, b10→5, b8→4, b7→4)
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    moe: bool = False
    mamba_hybrid: bool = False
    # --preset seq_len; config YAML / --opts can override (config wins)
    seq_len: int = 16384
    notes: str = ""


def _uniform_layers(n_layers: int, count: int) -> tuple[int, ...]:
    """Uniformly pick `count` layers in [1, n_layers-2] (DFLASH paper: 2nd to penultimate)."""
    lo, hi = 1, n_layers - 2
    step = (hi - lo) / (count - 1) if count > 1 else 0
    return tuple(round(lo + i * step) for i in range(count))


QWEN_PRESETS: dict[str, QwenPreset] = {
    "qwen38-27b": QwenPreset(
        name="qwen38-27b", model_path=f"{MODEL_ROOT}/Qwen3.8-27B",
        num_layers=64, hidden_size=5120, vocab_size=248320, arch="qwen3_5_text",
        block_size=8, decay_gamma=4, target_layer_ids=(5, 19, 33, 47, 61),
        num_target_layers=64, mamba_hybrid=True,
        notes="primary; target_layer_ids aligned with the reference DFLASH2 draft (serve/warm-start compatible)",
    ),
    "qwen35-122b": QwenPreset(
        name="qwen35-122b", model_path=f"{MODEL_ROOT}/../multimodal_raw_models/Qwen3.5-122B-A10B",
        num_layers=48, hidden_size=3072, vocab_size=248320, arch="qwen3_5_moe_text",
        block_size=16, decay_gamma=7, target_layer_ids=(1, 7, 14, 20, 26, 32, 39, 45),
        num_target_layers=48, moe=True,
        notes="MoE target: verify cost makes draft training often unprofitable — kept for reproducibility",
    ),
    "qwen3-8b": QwenPreset(
        name="qwen3-8b", model_path=f"{MODEL_ROOT}/Qwen3-8B",
        num_layers=36, hidden_size=4096, vocab_size=151936, arch="qwen3",
        block_size=8, decay_gamma=4, target_layer_ids=(2, 18, 33),
        num_target_layers=36,
        notes="smoke / single-GPU self-check",
    ),
    "qwen35-27b": QwenPreset(
        name="qwen35-27b", model_path=f"{MODEL_ROOT}/Qwen3.5-27B",
        num_layers=64, hidden_size=5120, vocab_size=248320, arch="qwen3_5_text",
        block_size=8, decay_gamma=4, target_layer_ids=(5, 19, 33, 47, 61),
        num_target_layers=64, mamba_hybrid=True,
        notes="Qwen3.5-27B, same architecture family as 3.8-27B",
    ),
    "qwen36-35b": QwenPreset(
        name="qwen36-35b", model_path=f"{MODEL_ROOT}/Qwen3.6-35B-A3B",
        num_layers=40, hidden_size=2048, vocab_size=248320, arch="qwen3_5_moe_text",
        block_size=8, decay_gamma=4, target_layer_ids=_uniform_layers(40, 8),
        num_target_layers=40, moe=True,
        notes="Qwen3.6-35B-A3B MoE; has an official DFLASH v1 draft (not warm-startable to 27B)",
    ),
    "qwen38-flash-next": QwenPreset(
        name="qwen38-flash-next", model_path=f"{MODEL_ROOT}/Qwen3.8-Flash-Next",
        num_layers=48, hidden_size=2560, vocab_size=248320, arch="qwen4_exp_text",
        block_size=8, decay_gamma=4, target_layer_ids=(1, 7, 14, 20, 26, 32, 39, 45),
        num_target_layers=48, moe=True, mamba_hybrid=True,
        seq_len=65536,
        notes="Flash-Next (qwen4_exp Mamba-hybrid+MoE) — experimental, untested in CI, seq_len 65536",
    ),
}


def get_preset(name: str) -> QwenPreset:
    if name not in QWEN_PRESETS:
        raise KeyError(
            f"unknown preset '{name}'. available: {sorted(QWEN_PRESETS)}"
        )
    return QWEN_PRESETS[name]


def preset_for_model(model_path: str) -> QwenPreset:
    """Match a preset by model path (for when --preset is omitted but --verifier is given)."""
    for p in QWEN_PRESETS.values():
        if model_path.rstrip("/") == p.model_path.rstrip("/"):
            return p
    raise KeyError(f"no preset matches '{model_path}'; pass --preset or --config explicitly")


def derive_target_layer_ids(num_layers: int, count: int) -> tuple[int, ...]:
    """Uniformly pick `count` target-layer features (used when not explicit)."""
    return _uniform_layers(num_layers, count)
