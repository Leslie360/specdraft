"""convert stage: speculators ckpt → sglang servable (PP4: dflash2 nested dflash_config)."""
from __future__ import annotations

import subprocess

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_convert


def run_convert(cfg: RunConfig, env: Env, dry_run: bool):
    cmd = cmd_convert(cfg, env)
    print("\n=== convert ===\n  " + " \\\n  ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env.environ(), check=True)
