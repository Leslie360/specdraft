"""RunConfig — full configuration for one run (preset resolution + YAML/.env loading + --opts overrides).

- `--config` accepts YAML or a legacy .env (KEY=VALUE shell format)
- `--opts k=v` arbitrary overrides, flat key (e.g. `train.epochs=10` → engine TrainConfig.flatten() dest)
- presets provide model defaults (layers / block / γ / vocab / target_layer_ids); config can override
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .presets import QwenPreset, get_preset


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a legacy shell .env (KEY=VALUE, ignores comments/export)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class RunConfig(BaseModel):
    # validate_assignment: a bad --opts value (e.g. seq_len=abc) must raise a
    # readable ValidationError, not silently stuff a string into the engine argv
    model_config = ConfigDict(validate_assignment=True)

    # --- target ---
    preset_name: str | None = None
    model_path: str | None = None          # explicit override of the preset

    # --- draft ---
    draft: str = "dflash"                   # dflash | dspark | dflash2
    num_layers: int = 5
    block_size: int | None = None           # None → from preset
    decay_gamma: int | None = None          # None → from preset
    target_layer_ids: list[int] | None = None  # None → from preset
    num_target_layers: int | None = None
    draft_vocab_size: int | None = None     # None → preset (DFLASH2 forces full vocab)
    max_anchors: int = 512
    seq_len: int = 8192

    # --- training ---
    lr: float = 3e-4
    epochs: int = 5
    optimizer: str = "muon_adamw"           # muon_adamw | adamw
    nproc: int = 4                          # DP device count
    fsdp: bool = False
    torch_compile: bool = False
    warm_start: str | None = None           # --from-pretrained path
    loss_fn: str | None = None              # None → per-draft default (dflash: ce+dpace; dspark/dflash2: see invoke)

    # --- data ---
    data_regen_jsonl: str | None = None     # raw regen input (prompt-only)
    data_pretok_jsonl: str | None = None    # pretokenize output
    data_prep: str | None = None            # prepare output (arrow)
    data_hs: str | None = None              # hidden-states directory
    data_token_freq: str | None = None

    # --- regen ---
    regen_endpoints: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8010/v1", "http://127.0.0.1:8011/v1"])
    regen_concurrency: int = 16
    regen_max_tokens: int = 4096
    regen_temperature: float = 0.7
    regen_top_p: float = 0.8
    regen_top_k: int = 20

    # --- hsextract ---
    extract_gpus: str = "0,1,2,3"
    extract_tp: int = 4
    extract_port: int = 8007

    # --- serve/bench ---
    serve_gpus: str = "0,1,2,3"
    serve_tp: int = 4
    serve_ports: list[int] = Field(default_factory=lambda: [8010])
    serve_ctx: int = 10248
    serve_mem_frac: float = 0.8
    serve_max_rq: int = 16
    spec_draft_model: str | None = None     # draft path to serve (defaults to convert output)
    bench_prompts: str | None = None
    bench_max_tokens: int = 128

    # --- output ---
    workdir: str = "./runs/default"
    ckpt_dir: str | None = None             # None → {workdir}/ckpt
    servable_dir: str | None = None         # None → {workdir}/servable

    # --- engine passthrough opts (map to TrainConfig.flatten() dest keys) ---
    opts: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        """Apply preset run defaults (seq_len / max_anchors / nproc) to fields left
        at their class default. Keeps RunConfig(preset_name=...) and load_config on
        the same semantics; explicit --opts overrides happen after construction and
        still win."""
        p = self.resolve_preset()
        if p is None:
            return
        if self.seq_len == RunConfig.model_fields["seq_len"].default and p.seq_len:
            self.seq_len = p.seq_len
        if self.max_anchors == RunConfig.model_fields["max_anchors"].default and p.max_anchors:
            self.max_anchors = p.max_anchors
        if self.nproc == RunConfig.model_fields["nproc"].default and p.nproc:
            self.nproc = p.nproc

    def resolve_preset(self) -> QwenPreset | None:
        if self.preset_name:
            return get_preset(self.preset_name)
        return None

    def resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        p = self.resolve_preset()
        if p:
            return p.model_path
        raise ValueError("either model_path or preset_name is required")

    def resolved_block_gamma(self) -> tuple[int, int]:
        p = self.resolve_preset()
        block = self.block_size or (p.block_size if p else 8)
        gamma = self.decay_gamma or (p.decay_gamma if p else 4)
        return block, gamma

    def resolved_target_layers(self) -> tuple[list[int], int]:
        p = self.resolve_preset()
        tl = self.target_layer_ids or (list(p.target_layer_ids) if p else None)
        ntl = self.num_target_layers or (p.num_target_layers if p else None)
        if tl is None or ntl is None:
            raise ValueError("target_layer_ids/num_target_layers must come from preset or config")
        return tl, ntl

    def resolved_vocab(self) -> int:
        p = self.resolve_preset()
        return self.draft_vocab_size or (p.vocab_size if p else 248320)

    def resolved_ckpt(self) -> str:
        return self.ckpt_dir or f"{self.workdir}/ckpt"

    def resolved_servable(self) -> str:
        return self.servable_dir or f"{self.workdir}/servable"


def load_config(config_path: str | None, preset: str | None, opts: list[str] | None) -> RunConfig:
    """Assemble a RunConfig from --config (YAML or .env) + --preset + --opts."""
    raw: dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if p.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            raw = dict(_load_env_file(p))  # .env: all strings, some need casting on RunConfig
    if preset:
        raw["preset_name"] = preset
    cfg = RunConfig(**raw)
    # preset run defaults (seq_len/max_anchors/nproc for fields left at class
    # default) are applied by RunConfig.model_post_init — both --preset and
    # --config paths share it; explicit --opts below still win.

    for o in opts or []:
        if "=" not in o:
            raise ValueError(f"--opts must be k=v, got: {o}")
        k, v = o.split("=", 1)
        _apply_opt(cfg, k, v)
    return cfg


def _apply_opt(cfg: RunConfig, key: str, value: str):
    """--opts override: try a RunConfig field first, else pass through to the opts dict."""
    # simple bool/int/float cast + list fields (comma-separated → list)
    def cast(s: str):
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    if key in RunConfig.model_fields:
        field_type = RunConfig.model_fields[key].annotation
        # list fields: comma-separated
        if "list" in str(field_type).lower() and "," in value:
            setattr(cfg, key, [cast(x.strip()) for x in value.split(",")])
        else:
            setattr(cfg, key, cast(value))
    else:
        cfg.opts[key] = cast(value)
