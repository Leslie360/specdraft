"""bench stage: send prompts, read spec_accept_length/rate (reuses bench_accept.py)."""
from __future__ import annotations

import subprocess

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_bench


def run_bench(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.bench_prompts:
        if dry_run:
            print("\n=== bench (skipped: requires bench_prompts) ===")
            return
        raise ValueError("bench requires bench_prompts (a json array file)")
    for port in cfg.serve_ports:
        cmd = cmd_bench(cfg, env, port=port)
        print("\n=== bench :%d ===\n  " % port + " \\\n  ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, env=env.environ(), check=True)
