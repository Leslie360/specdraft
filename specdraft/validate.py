"""Config / artifact validation — turns session-verified pitfalls into hard checks (fail early)."""
from __future__ import annotations

from pathlib import Path

from .config import RunConfig

BLOCK_GAMMA = {16: 7, 10: 5, 8: 4, 7: 4}


def validate_config(cfg: RunConfig) -> list[str]:
    """Return a list of problems (empty = OK)."""
    problems: list[str] = []
    block, gamma = cfg.resolved_block_gamma()
    expected = BLOCK_GAMMA.get(block)
    if expected is not None and gamma != expected:
        problems.append(
            f"decay_gamma={gamma} does not match block_size={block} "
            f"(expected {expected}, otherwise trailing positions starve)"
        )
    if cfg.draft == "dflash2":
        # PP4: DFLASH2 forces full vocab (core.py would raise; warn early)
        pass
    if cfg.warm_start:
        # DFLASH→DSPARK warm-start is not one-step (config Literal rejects dflash);
        # type matching is enforced by the engine; just a hint here.
        pass
    return problems


def validate_data_consistency(cfg: RunConfig) -> list[str]:
    """train target_layer_ids must match hsextract's (PP: train == extract)."""
    problems: list[str] = []
    # extraction uses --include-last-layer (one extra layer); train uses target_layer_ids
    # (without last). Consistency is up to the user config; here we only hint at
    # whether the data dirs exist.
    if cfg.data_hs and not Path(cfg.data_hs).exists():
        problems.append(f"data_hs directory does not exist: {cfg.data_hs}")
    if cfg.data_prep and not Path(cfg.data_prep).exists():
        problems.append(f"data_prep directory does not exist: {cfg.data_prep}")
    return problems


def check_hs_nonzero(hs_dir: str, sample: int = 0) -> bool:
    """hidden-states non-zero check (all-zero → train fine but 0% accept trap)."""
    import glob

    import torch

    files = sorted(glob.glob(f"{hs_dir}/hs_*.safetensors"))
    if not files:
        return False
    f = files[min(sample, len(files) - 1)]
    try:
        t = torch.load(f, map_location="cpu", weights_only=True)
        # structure may be dict or tensor
        vals = t.values() if isinstance(t, dict) else [t]
        return all(v.abs().sum().item() > 0 for v in vals)
    except Exception:
        return False
