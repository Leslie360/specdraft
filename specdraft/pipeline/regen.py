"""regen stage: on-policy response regeneration (PP6: multi-instance + sharding + resume).

Input:  prompt-only jsonl (data_regen_jsonl)
Output: regen jsonl (the regenerated responses for those prompts)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_regen


def _split_input(path: str, n: int, workdir: str) -> list[str]:
    """Split the input jsonl into n roughly-equal parts (one per parallel instance)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    out = []
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    per = (len(rows) + n - 1) // n
    for i in range(n):
        part = rows[i * per:(i + 1) * per]
        p = wd / f"regen_part_{i}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.append(str(p))
    return out


def _regen_output_path(cfg: RunConfig) -> Path:
    """Where the regen stage writes its output (the on-policy regen jsonl).

    This MUST be the file pretokenize reads: <data_pretok stem>_regen.jsonl keeps
    the regen output next to the pretok file without colliding with it. (The old
    parallel path merged into data_pretok_jsonl itself, which pretokenize then
    overwrote with 0 rows when fed the prompt-only input — a silent pipeline
    data-flow break only visible on a real run.)
    """
    if cfg.data_pretok_jsonl:
        p = Path(cfg.data_pretok_jsonl)
        return p.with_name(p.stem + "_regen.jsonl")
    in_path = Path(cfg.data_regen_jsonl)
    return in_path.with_name(in_path.stem + "_regen.jsonl")


def run_regen(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_regen_jsonl:
        if dry_run:
            print("\n=== regen (skipped: requires data_regen_jsonl) ===")
            return
        raise ValueError("regen requires data_regen_jsonl (prompt-only jsonl)")
    out = _regen_output_path(cfg)
    eps = cfg.regen_endpoints
    if len(eps) == 1:
        cmd = cmd_regen(cfg, env, endpoint=eps[0],
                        input_jsonl=cfg.data_regen_jsonl,
                        output_jsonl=str(out))
        print("\n=== regen (single instance) ===\n  " + " \\\n  ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, env=env.environ(), check=True)
        return

    # parallel multi-instance (PP6): shard → one part per instance → merge
    parts = _split_input(cfg.data_regen_jsonl, len(eps), f"{cfg.workdir}/regen_parts")
    cmds = []
    for i, (ep, part) in enumerate(zip(eps, parts)):
        out_i = f"{Path(cfg.workdir)}/regen_out_{i}.jsonl"
        cmds.append(cmd_regen(cfg, env, endpoint=ep, input_jsonl=part, output_jsonl=out_i))
    print("\n=== regen (parallel %d instances) ===" % len(cmds))
    for c in cmds:
        print("  " + " \\\n  ".join(c))
    if dry_run:
        return
    procs = [subprocess.Popen(c, env=env.environ()) for c in cmds]
    for p in procs:
        p.wait()
    # merge into the shared regen output path (what pretokenize consumes)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for i in range(len(cmds)):
            out_i = f"{Path(cfg.workdir)}/regen_out_{i}.jsonl"
            for line in open(out_i, encoding="utf-8"):
                fout.write(line)
    print(f"regen merged -> {out}")
