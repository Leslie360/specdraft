"""Environment resolution: interpreters / speculators repo path / subprocess env.

Known pitfalls (verified on real sessions):
- The speculators engine scripts live in the repo's scripts/ and need both
  src/ and hs_connectors/src on PYTHONPATH → point SPECDRAFT_SPECULATORS_REPO at
  a clone of https://github.com/vllm-project/speculators
- Old driver → prepend a cuda compat dir to LD_LIBRARY_PATH
  (overridable via SPECDRAFT_CUDA_COMPAT)
- vLLM custom_all_reduce crashes on GDN (Mamba-hybrid) → --disable-custom-all-reduce
  + mp backend
- Never set VLLM_BATCH_INVARIANT=1 (conflicts with GDN_ATTN)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "vendor"


def _require_env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        # ValueError (not RuntimeError): this is a user-environment setup error, so
        # cli.main's readable-error handler catches it — no bare traceback
        raise ValueError(
            f"missing env var {name} — see README 'Environment' section"
        )
    return v


def speculators_repo() -> Path:
    """Path to a vllm-speculators clone (where the engine scripts live)."""
    return Path(_require_env("SPECDRAFT_SPECULATORS_REPO"))


def default_train_python() -> str:
    """Training interpreter with vllm / torch / transformers / speculators."""
    return _require_env("SPECDRAFT_TRAIN_PYTHON")


def default_sglang_python() -> str:
    """Serve interpreter with sglang (defaults to python3 on PATH)."""
    return os.environ.get("SPECDRAFT_SGLANG_PYTHON", "python3")


def optional_proxy() -> str | None:
    """Optional proxy (e.g. for HF downloads); unset disables it."""
    return os.environ.get("SPECDRAFT_PROXY") or None


@dataclass(frozen=True)
class Env:
    """Interpreters and subprocess environment for one run."""

    python: str = ""           # training/engine interpreter (defaults to SPECDRAFT_TRAIN_PYTHON)
    sglang_python: str = ""    # sglang serve interpreter (defaults to SPECDRAFT_SGLANG_PYTHON or PATH)

    def __post_init__(self):
        if not self.python:
            object.__setattr__(self, "python", default_train_python())
        if not self.sglang_python:
            object.__setattr__(self, "sglang_python", default_sglang_python())

    # --- script paths ---

    def script(self, name: str) -> str:
        """Engine script inside the speculators repo (scripts/<name>)."""
        return str(self.speculators_dir() / "scripts" / name)

    def vendored(self, name: str) -> str:
        """Helper script under this repo's vendor/."""
        return str(VENDOR / name)

    def speculators_dir(self) -> Path:
        return speculators_repo()

    # --- subprocess environment ---

    def pythonpath(self) -> str:
        sp = self.speculators_dir()
        parts = [str(sp / "src"), str(sp / "hs_connectors" / "src")]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        return ":".join(parts)

    def environ(self, *, for_vllm: bool = False, for_sglang: bool = False) -> dict[str, str]:
        """Subprocess environment (pass at subprocess launch)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = self.pythonpath()
        cuda_compat = os.environ.get("SPECDRAFT_CUDA_COMPAT", "/usr/local/cuda-13.0/compat")
        if Path(cuda_compat).exists():
            env["LD_LIBRARY_PATH"] = cuda_compat + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
            )
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        # PP1: never set VLLM_BATCH_INVARIANT (conflicts with GDN)
        env.pop("VLLM_BATCH_INVARIANT", None)
        if for_sglang:
            env["SGLANG_DISABLE_OVERLAP_SCHEDULER"] = env.get("SGLANG_DISABLE_OVERLAP_SCHEDULER", "")
        return env

    def use_proxy(self, env: dict[str, str] | None = None) -> dict[str, str]:
        """Add proxy env vars when SPECDRAFT_PROXY is set; otherwise passthrough."""
        env = dict(env or os.environ)
        proxy = optional_proxy()
        if proxy:
            env["https_proxy"] = proxy
            env["http_proxy"] = proxy
        return env
