"""train stage: torch.distributed.run to train the draft (DP; argv per draft type assembled in invoke)."""
from __future__ import annotations

import subprocess

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_train
from ..validate import validate_config, validate_data_consistency


def run_train(cfg: RunConfig, env: Env, dry_run: bool):
    probs = validate_config(cfg)
    # data paths/dir existence are environment checks: irrelevant in dry-run (argv
    # must be inspectable without data), hard-checked on a real run
    if not dry_run:
        if not cfg.data_prep or not cfg.data_hs:
            probs.append("train requires data_prep + data_hs (set --config / --opts)")
        probs += validate_data_consistency(cfg)
    if probs:
        raise ValueError("train config validation failed:\n  " + "\n  ".join(probs))
    cmd = cmd_train(cfg, env)
    print("\n=== train (%s) ===\n  " % cfg.draft + " \\\n  ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env.environ(), check=True)
