#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 specdraft contributors

"""Convert speculators train.py outputs to sglang-servable drafts.

Generic:
  - DFLASH : architectures=["DFlashDraftModel"]
  - DSPARK : architectures=["DSparkDraftModel"] (with markov/confidence heads)
  - drops embed_tokens.weight / lm_head.weight uniformly (shared with the target at runtime)

Usage:
  python 04_convert_servable.py <ckpt_dir> <out_dir> [--arch dspark] [--num-target-layers 48]
"""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP_KEYS = {"embed_tokens.weight", "lm_head.weight"}

ARCH_MAP = {
    "dflash": "DFlashDraftModel",
    "dspark": "DSparkDraftModel",
    "dflash2": "DFlash2DraftModel",  # port of PR#1006 (2026-08-25); supported
}


def build_servable_config(src_cfg: dict, arch: str, num_target_layers: int) -> dict:
    tlc = src_cfg.get("transformer_layer_config", src_cfg)
    cfg = {
        "architectures": [ARCH_MAP[arch]],
        "model_type": tlc.get("model_type", "qwen3"),
        # drop auto_map: sglang registers by architectures (auto_map pulls in modules missing from sglang_venv)
        "dtype": "bfloat16",
        "transformers_version": src_cfg.get("transformers_version", "5.13.0"),
        # flat HF transformer fields
        "attention_bias": tlc.get("attention_bias", False),
        "attention_dropout": tlc.get("attention_dropout", 0.0),
        "bos_token_id": tlc.get("bos_token_id"),
        "eos_token_id": tlc.get("eos_token_id"),
        "head_dim": tlc.get("head_dim"),
        "hidden_act": tlc.get("hidden_act", "silu"),
        "hidden_size": tlc["hidden_size"],
        "initializer_range": tlc.get("initializer_range", 0.02),
        "intermediate_size": tlc.get("intermediate_size"),
        "layer_types": tlc.get("layer_types"),
        "max_position_embeddings": tlc.get("max_position_embeddings"),
        "max_window_layers": tlc.get("max_window_layers"),
        "num_attention_heads": tlc["num_attention_heads"],
        "num_hidden_layers": tlc["num_hidden_layers"],
        "num_key_value_heads": tlc["num_key_value_heads"],
        "pad_token_id": tlc.get("pad_token_id"),
        "rms_norm_eps": tlc.get("rms_norm_eps"),
        "rope_parameters": tlc.get("rope_parameters"),
        "sliding_window": tlc.get("sliding_window"),
        "tie_word_embeddings": tlc.get("tie_word_embeddings", False),
        "use_cache": tlc.get("use_cache", True),
        "use_sliding_window": tlc.get("use_sliding_window", True),
        "vocab_size": src_cfg.get("draft_vocab_size", tlc.get("vocab_size")),
        # speculator fields
        "block_size": src_cfg.get("block_size"),
        "mask_token_id": src_cfg.get("mask_token_id"),
        "target_layer_ids": src_cfg.get("aux_hidden_state_layer_ids"),
        "num_target_layers": num_target_layers,
    }
    # DSpark-only head fields
    if arch == "dspark":
        cfg.update({
            "markov_rank": src_cfg.get("markov_rank"),
            "markov_head_type": src_cfg.get("markov_head_type", "vanilla"),
            "enable_confidence_head": src_cfg.get("enable_confidence_head", False),
            "confidence_head_with_markov": src_cfg.get("confidence_head_with_markov", False),
            "sample_from_anchor": src_cfg.get("sample_from_anchor"),
        })
    # DFlash2-only: nested dflash_config (read by sglang DFlash2DraftModel; active when selector_rank is truthy)
    if arch == "dflash2":
        cfg["dflash_config"] = {
            "block_size": src_cfg.get("block_size"),
            "conv_kernel_size": src_cfg.get("conv_kernel_size", 2),
            "conv_group_size": src_cfg.get("conv_group_size", 16),
            "selector_rank": src_cfg.get("selector_rank", 256),
            "selector_top_k": src_cfg.get("selector_top_k", 16),
            "mask_token_id": src_cfg.get("mask_token_id"),
            "target_layer_ids": src_cfg.get("aux_hidden_state_layer_ids"),
            "num_target_layers": num_target_layers,
        }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="speculators train.py save dir (config.json+model.safetensors)")
    ap.add_argument("out", type=Path, help="output directory for the servable draft")
    ap.add_argument("--arch", choices=list(ARCH_MAP), default="dspark")
    ap.add_argument("--num-target-layers", type=int, default=48, help="target num_hidden_layers")
    args = ap.parse_args()

    src_cfg = json.load(open(args.ckpt / "config.json"))
    weights = load_file(args.ckpt / "model.safetensors")
    kept = {k: v for k, v in weights.items() if k not in DROP_KEYS}
    cfg = build_servable_config(src_cfg, args.arch, args.num_target_layers)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(kept, args.out / "model.safetensors")
    json.dump(cfg, open(args.out / "config.json", "w"), indent=2)

    print(f"source keys: {len(weights)}, dropped: {sorted(weights.keys() & DROP_KEYS)}")
    print(f"kept: {len(kept)}, arch={cfg['architectures']}")
    print(f"block_size={cfg['block_size']} target_layer_ids={cfg['target_layer_ids']} "
          f"num_target_layers={cfg['num_target_layers']}")
    if args.arch == "dspark":
        print(f"markov_rank={cfg['markov_rank']} markov_head_type={cfg['markov_head_type']} "
              f"enable_confidence_head={cfg['enable_confidence_head']}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
