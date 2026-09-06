#!/usr/bin/env python3
"""specdraft CLI — entry point for speculative-draft training on the Qwen family.

Usage:
  specdraft dflash  --preset qwen38-27b --stage all --dry-run
  specdraft dspark  --config runs/qwen38_27b.yaml --stage train --opts train.epochs=10
  specdraft dflash2 --preset qwen38-27b --stage all
"""
from __future__ import annotations

import argparse

from .config import load_config
from .env import Env
from .pipeline import ORDER, run_stage, run_all


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="specdraft",
                                 description="speculative-draft training pipeline (specdraft)")
    ap.add_argument("draft", choices=["dflash", "dspark", "dflash2"])
    ap.add_argument("--preset", default=None, help="Qwen preset name (e.g. qwen38-27b)")
    ap.add_argument("--config", default=None, help="YAML or legacy .env config file")
    ap.add_argument("--stage", default="all",
                    help="all or a single stage: " + ",".join(ORDER))
    ap.add_argument("--opts", action="append", default=[], metavar="K=V",
                    help="arbitrary overrides (e.g. train.epochs=10)")
    ap.add_argument("--dry-run", action="store_true", help="print commands only, do not execute")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.preset, args.opts)
    cfg.draft = args.draft
    env = Env()

    from .validate import validate_config
    probs = validate_config(cfg)
    if probs:
        print("⚠ config validation reminders:")
        for p in probs:
            print("  -", p)

    print(f"target: {cfg.draft} @ {cfg.resolved_model_path()}")
    print(f"block={cfg.resolved_block_gamma()[0]} γ={cfg.resolved_block_gamma()[1]} "
          f"layers={cfg.num_layers} target={cfg.resolved_target_layers()[0]}")
    print(f"workdir={cfg.workdir} dry_run={args.dry_run}")

    if args.stage == "all":
        run_all(cfg, env, args.dry_run)
    else:
        run_stage(args.stage, cfg, env, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] Commands printed above, nothing executed. Drop --dry-run to run for real.")


if __name__ == "__main__":
    main()
