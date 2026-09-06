"""pretokenize stage: natural-language regen → speculator-format rows (input_ids + loss_mask).

PP2 render-endpoint workaround: prepare skips render when rows carry input_ids+loss_mask.
PP3: BatchEncoding fails isinstance(dict) + missing assistant_tokens_mask → use the last <|im_start|>.
Calls vendor/pre_tokenize.py (built into this repo).
"""
from __future__ import annotations

import subprocess

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_pretokenize


def run_pretokenize(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_regen_jsonl:
        raise ValueError("pretokenize requires data_regen_jsonl")
    if not cfg.data_pretok_jsonl:
        raise ValueError("pretokenize requires data_pretok_jsonl")
    cmd = cmd_pretokenize(cfg, env)
    print("\n=== pretokenize ===\n  " + " \\\n  ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env.environ(), check=True)
