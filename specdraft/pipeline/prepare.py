"""prepare stage: pretokenize rows → tokenized arrow prep (PP2: no render-endpoint)."""
from __future__ import annotations

import subprocess

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_prepare


def run_prepare(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_pretok_jsonl:
        raise ValueError("prepare requires data_pretok_jsonl")
    if not cfg.data_prep:
        raise ValueError("prepare requires data_prep")
    cmd = cmd_prepare(cfg, env)
    print("\n=== prepare ===\n  " + " \\\n  ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env.environ(), check=True)
