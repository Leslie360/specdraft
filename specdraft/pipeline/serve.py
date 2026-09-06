"""serve stage: sglang serve (PP5: multi-instance / TP assignment; PP1-compatible with DFLASH2 drafts)."""
from __future__ import annotations

import subprocess
import time

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_serve


def run_serve(cfg: RunConfig, env: Env, dry_run: bool):
    # server assignment: single instance by default; multi-instance uses (gpus, port) pairs
    if not cfg.serve_ports:
        raise ValueError("serve requires serve_ports")
    ports = cfg.serve_ports
    gpus_list = [cfg.serve_gpus] if len(ports) == 1 else _split_gpus(cfg.serve_gpus, len(ports))
    tps = [cfg.serve_tp] * len(ports)
    cmds = []
    for port, gpus, tp in zip(ports, gpus_list, tps):
        cmds.append(cmd_serve(cfg, env, port=port, gpus=gpus, tp=tp))
    print(f"\n=== serve ({len(cmds)} instances) ===")
    for c in cmds:
        print("  " + " \\\n  ".join(c))
    if dry_run:
        return
    procs = []
    for c, gpus in zip(cmds, gpus_list):
        envd = env.environ(for_sglang=True)
        envd["CUDA_VISIBLE_DEVICES"] = gpus
        procs.append(subprocess.Popen(c, env=envd))
    # wait for all health OK
    for port in ports:
        _wait_health(port)
    print("serve ready:", ports)


def _split_gpus(gpus: str, n: int) -> list[str]:
    ids = [g for g in gpus.split(",") if g]
    if len(ids) < n:
        raise ValueError(f"GPU count {len(ids)} < instance count {n}")
    return [",".join(ids[i::n]) for i in range(n)]


def _wait_health(port: int, timeout_s: int = 600):
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"serve :{port} not ready")
