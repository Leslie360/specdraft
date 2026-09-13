"""pretokenize stage: natural-language regen → speculator-format rows (input_ids + loss_mask).

PP2 render-endpoint workaround: prepare skips render when rows carry input_ids+loss_mask.
PP3: BatchEncoding fails isinstance(dict) + missing assistant_tokens_mask → use the last <|im_start|>.
Calls vendor/pre_tokenize.py (built into this repo).

Input contract: the regen stage's OUTPUT (on-policy jsonl: system/user + generated
assistant turns) — NOT the prompt-only data_regen_jsonl. When the regen output file
exists it wins; data_regen_jsonl only serves as an explicit override for runs whose
regen jsonl was produced out-of-band (named exactly as configured).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_pretokenize
from .regen import _regen_output_path


def run_pretokenize(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_pretok_jsonl:
        if dry_run:
            print("\n=== pretokenize (skipped: requires data_pretok_jsonl) ===")
            return
        raise ValueError("pretokenize requires data_pretok_jsonl")

    regen_out = _regen_output_path(cfg)
    input_path = str(regen_out) if regen_out.is_file() else cfg.data_regen_jsonl
    if not input_path or not Path(input_path).is_file():
        if dry_run:
            print("\n=== pretokenize (skipped: requires the regen output "
                  f"{regen_out} or data_regen_jsonl) ===")
            return
        raise ValueError(
            f"pretokenize input not found: expected the regen output {regen_out} "
            "(run the regen stage first) or set data_regen_jsonl to an existing "
            "on-policy regen jsonl")

    cmd = cmd_pretokenize(cfg, env, input_jsonl=input_path)
    print("\n=== pretokenize ===\n  " + " \\\n  ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env.environ(), check=True)
