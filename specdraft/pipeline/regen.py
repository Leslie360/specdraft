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


def run_regen(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_regen_jsonl:
        if dry_run:
            print("\n=== regen (skipped: requires data_regen_jsonl) ===")
            return
        raise ValueError("regen requires data_regen_jsonl (prompt-only jsonl)")
    eps = cfg.regen_endpoints
    if len(eps) == 1:
        # default output: <input-stem>_regen.jsonl next to the input; the explicit
        # data_pretok_jsonl (if set) is only a stem hint, never a None deref
        in_path = Path(cfg.data_regen_jsonl)
        default_out = in_path.with_name(in_path.stem + "_regen.jsonl")
        out = (Path(cfg.data_pretok_jsonl).with_name(Path(cfg.data_pretok_jsonl).stem + "_regen.jsonl")
               if cfg.data_pretok_jsonl else default_out)
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
        out = f"{Path(cfg.workdir)}/regen_out_{i}.jsonl"
        cmds.append(cmd_regen(cfg, env, endpoint=ep, input_jsonl=part, output_jsonl=out))
    print("\n=== regen (parallel %d instances) ===" % len(cmds))
    for c in cmds:
        print("  " + " \\\n  ".join(c))
    if dry_run:
        return
    procs = [subprocess.Popen(c, env=env.environ()) for c in cmds]
    for p in procs:
        p.wait()
    # merge
    merged = cfg.data_pretok_jsonl or f"{Path(cfg.workdir)}/regen_merged.jsonl"
    with open(merged, "w", encoding="utf-8") as fout:
        for i in range(len(cmds)):
            out = f"{Path(cfg.workdir)}/regen_out_{i}.jsonl"
            for line in open(out, encoding="utf-8"):
                fout.write(line)
    print(f"regen merged -> {merged}")
