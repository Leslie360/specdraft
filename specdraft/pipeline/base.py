"""Stage contract: ordered registration, validate_in/out, dry-run support."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from ..config import RunConfig
from ..env import Env


@dataclass
class Stage:
    name: str
    run: Callable[[RunConfig, Env, bool], None]  # (cfg, env, dry_run)
    validate_in: Callable[[RunConfig], list[str]] | None = None
    validate_out: Callable[[RunConfig], list[str]] | None = None


def _run_cmd(cmd: list[str], env: Env, dry_run: bool, label: str):
    """Execute or print a command."""
    envdict = env.environ()
    print(f"\n=== {label} ===\n  " + " \\\n  ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, env=envdict, check=True)


# ---- stage implementations (one file each; registered here) ----
from . import bench, convert, hsextract, prepare, pretokenize, regen, serve, train  # noqa: E402,F401

STAGES: dict[str, Stage] = {
    "regen": Stage("regen", regen.run_regen),
    "pretokenize": Stage("pretokenize", pretokenize.run_pretokenize),
    "prepare": Stage("prepare", prepare.run_prepare),
    "hsextract": Stage("hsextract", hsextract.run_hsextract),
    "train": Stage("train", train.run_train),
    "convert": Stage("convert", convert.run_convert),
    "serve": Stage("serve", serve.run_serve),
    "bench": Stage("bench", bench.run_bench),
}

ORDER = ["regen", "pretokenize", "prepare", "hsextract", "train", "convert", "serve", "bench"]


def run_stage(name: str, cfg: RunConfig, env: Env, dry_run: bool):
    if name not in STAGES:
        raise KeyError(f"unknown stage '{name}'. available: {ORDER}")
    st = STAGES[name]
    if st.validate_in:
        probs = st.validate_in(cfg)
        if probs:
            raise ValueError(f"[{name}] pre-check failed: {probs}")
    st.run(cfg, env, dry_run)
    if st.validate_out:
        probs = st.validate_out(cfg)
        if probs:
            raise ValueError(f"[{name}] artifact check failed: {probs}")


def run_all(cfg: RunConfig, env: Env, dry_run: bool, stages: list[str] | None = None):
    sel = stages or ORDER
    for s in sel:
        if s not in ORDER:
            raise KeyError(f"unknown stage '{s}'")
    for s in sel:
        run_stage(s, cfg, env, dry_run)
