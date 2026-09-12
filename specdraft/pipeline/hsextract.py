"""hsextract stage: launch vLLM (target-layer-ids) to extract hidden-states → disk (PP1 flags built in)."""
from __future__ import annotations

import subprocess
import time

from ..config import RunConfig
from ..env import Env
from ..invoke import cmd_hsextract, cmd_hsoffline


def _wait_health(url: str, timeout_s: int = 360, dry_run: bool = False) -> bool:
    if dry_run:
        return True
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def run_hsextract(cfg: RunConfig, env: Env, dry_run: bool):
    if not cfg.data_prep or not cfg.data_hs:
        if dry_run:
            print("\n=== hsextract (skipped: requires data_prep + data_hs) ===")
            return
        raise ValueError("hsextract requires data_prep + data_hs")
    # Stage A: launch vLLM (with target_layer_ids + include-last-layer)
    vcmd = cmd_hsextract(cfg, env)
    print("\n=== hsextract A: vLLM ===\n  " + " \\\n  ".join(vcmd))
    if dry_run:
        print("\n=== hsextract B: offline hs extraction ===\n  " + " \\\n  ".join(cmd_hsoffline(cfg, env)))
        return
    envdict = env.environ()
    proc = subprocess.Popen(vcmd, env=envdict)
    try:
        ok = _wait_health(f"http://127.0.0.1:{cfg.extract_port}")
        if not ok:
            raise RuntimeError(f"vLLM failed to start: :{cfg.extract_port}")
        # Stage B: offline hs extraction
        ocmd = cmd_hsoffline(cfg, env)
        print("\n=== hsextract B: offline hs extraction ===\n  " + " \\\n  ".join(ocmd))
        subprocess.run(ocmd, env=envdict, check=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print(f"hsextract done -> {cfg.data_hs}")
