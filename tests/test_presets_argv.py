"""Preset-wide argv smoke: per-preset values must land in the right flags on BOTH
config paths (--preset table and presets/*.yaml), no engine needed.

- EXPECTED is hand-transcribed from presets/*.yaml — an independent oracle, never
  imported from specdraft.
- _cfg() resolves through the preset TABLE (RunConfig(preset_name=...)), so the
  table itself is exercised; test_preset_table_matches_yaml cross-checks table vs
  YAML so the two paths cannot drift.
- The remaining tests assert the assembled argv per preset and the four data/
  serving stage builders (pretokenize/prepare/hsoffline/regen/bench).
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from specdraft.cli import main as cli_main  # noqa: E402
from specdraft.config import RunConfig, load_config  # noqa: E402
from specdraft.env import Env  # noqa: E402
from specdraft.invoke import (  # noqa: E402
    TRAIN,
    cmd_bench,
    cmd_convert,
    cmd_hsextract,
    cmd_hsoffline,
    cmd_pretokenize,
    cmd_prepare,
    cmd_regen,
    cmd_serve,
    cmd_train,
)
from specdraft.presets import QWEN_PRESETS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

EXPECTED = {
    "qwen3-8b": dict(block=8, gamma=4, vocab=151936, seq_len=4096, anchors=512,
                     nproc=1, layers=[2, 18, 33], ntl=36),
    "qwen35-122b": dict(block=16, gamma=7, vocab=248320, seq_len=8192, anchors=224,
                        nproc=8, layers=[1, 7, 14, 20, 26, 32, 39, 45], ntl=48),
    "qwen35-27b": dict(block=8, gamma=4, vocab=248320, seq_len=8192, anchors=512,
                       nproc=4, layers=[5, 19, 33, 47, 61], ntl=64),
    "qwen36-35b": dict(block=8, gamma=4, vocab=248320, seq_len=8192, anchors=224,
                       nproc=4, layers=[1, 6, 12, 17, 22, 27, 33, 38], ntl=40),
    "qwen38-27b": dict(block=8, gamma=4, vocab=248320, seq_len=8192, anchors=512,
                       nproc=4, layers=[5, 19, 33, 47, 61], ntl=64),
    "qwen38-flash-next": dict(block=8, gamma=4, vocab=248320, seq_len=65536, anchors=512,
                              nproc=8, layers=[1, 7, 14, 20, 26, 32, 39, 45], ntl=48),
}

PRESETS = sorted(QWEN_PRESETS)
assert len(PRESETS) == 6 and set(PRESETS) == set(EXPECTED)


def _cfg(preset: str, draft: str = "dflash", **overrides) -> RunConfig:
    """Resolve through the preset TABLE (exercises presets.py, not the YAML)."""
    cfg = RunConfig(preset_name=preset, data_prep="/tmp/prep", data_hs="/tmp/hs",
                    **overrides)
    cfg.draft = draft
    return cfg


def _flags(cmd: list[str], positional: set[str] = frozenset()) -> dict[str, list[str]]:
    """Map each --flag to its value tokens (up to the next --flag).

    Tokens in `positional` are script paths / subcommand literals — neither a flag
    nor a flag value (e.g. torchrun's engine script positional).
    """
    out: dict[str, list[str]] = {}
    cur = None
    for tok in cmd:
        if tok.startswith("--"):
            cur = tok
            out.setdefault(cur, [])
        elif cur is not None and tok not in positional:
            out[cur].append(tok)
    return out


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_table_matches_yaml(preset):
    """The python table and presets/<name>.yaml must agree — otherwise --preset
    and --config runs silently differ (this exact drift shipped once: the
    qwen36-35b YAML had wrong target layers, and seq_len diverged on 5 presets)."""
    exp = EXPECTED[preset]
    p = QWEN_PRESETS[preset]
    y = yaml.safe_load((REPO / "presets" / f"{preset}.yaml").read_text(encoding="utf-8"))
    assert y["preset_name"] == preset
    assert list(p.target_layer_ids) == y["target_layer_ids"] == exp["layers"]
    assert p.block_size == y["block_size"] == exp["block"]
    assert p.decay_gamma == y["decay_gamma"] == exp["gamma"]
    assert p.vocab_size == y["draft_vocab_size"] == exp["vocab"]
    assert p.num_target_layers == y["num_target_layers"] == exp["ntl"]
    assert p.seq_len == y["seq_len"] == exp["seq_len"]
    assert p.max_anchors == y["max_anchors"] == exp["anchors"]
    assert p.nproc == y["nproc"] == exp["nproc"]
    # both config paths resolve to the same effective run config
    via_table = RunConfig(preset_name=preset)
    via_yaml = load_config(str(REPO / "presets" / f"{preset}.yaml"), None, [])
    for field in ("seq_len", "max_anchors", "nproc"):
        assert getattr(via_table, field) == getattr(via_yaml, field), field


@pytest.mark.parametrize("draft", ["dflash", "dspark", "dflash2"])
@pytest.mark.parametrize("preset", PRESETS)
def test_train_argv_smoke(preset, draft):
    exp = EXPECTED[preset]
    flags = _flags(cmd_train(_cfg(preset, draft), Env()),
                   positional={Env().script(TRAIN)})
    assert flags["--speculator-type"] == [draft]
    assert flags["--block-size"] == [str(exp["block"])]
    assert flags["--dflash-decay-gamma"] == [str(exp["gamma"])]
    assert flags["--max-anchors"] == [str(exp["anchors"])]
    assert flags["--draft-vocab-size"] == [str(exp["vocab"])]
    assert flags["--total-seq-len"] == [str(exp["seq_len"])]
    assert flags["--target-layer-ids"] == [str(x) for x in exp["layers"]]
    assert flags["--nproc_per_node"] == [str(exp["nproc"])]
    assert flags["--on-missing"] == ["raise"]
    # per-draft loss/head flags (PP1/PP4 semantics)
    if draft == "dflash":
        assert flags["--per-position-loss-weight"] == ["dpace"]
    elif draft == "dspark":
        assert flags["--markov-rank"] == ["256"]
        assert "--enable-confidence-head" in flags
    else:  # dflash2 (PP4: full argv + full vocab)
        assert flags["--selector-rank"] == ["256"]
        assert flags["--conv-kernel-size"] == ["2"]
        assert "--sliding-window-non-causal" in flags
        assert flags["--draft-vocab-size"] == [str(exp["vocab"])]


@pytest.mark.parametrize("preset", PRESETS)
def test_convert_argv_smoke(preset):
    exp = EXPECTED[preset]
    flags = _flags(cmd_convert(_cfg(preset), Env()))
    assert flags["--arch"] == ["dflash"]
    assert flags["--num-target-layers"] == [str(exp["ntl"])]


@pytest.mark.parametrize("preset", PRESETS)
def test_serve_argv_smoke(preset):
    cfg = _cfg(preset)
    flags = _flags(cmd_serve(cfg, Env(), port=8010, gpus="0,1", tp=2))
    assert flags["--speculative-algorithm"] == ["DFLASH"]
    assert flags["--speculative-draft-model-path"] == [cfg.resolved_servable()]
    assert flags["--speculative-num-draft-tokens"] == ["8"]
    assert flags["--tp"] == ["2"]
    assert flags["--port"] == ["8010"]
    assert flags["--context-length"] == [str(cfg.serve_ctx)]
    assert flags["--mem-fraction-static"] == [str(cfg.serve_mem_frac)]
    assert flags["--max-running-requests"] == [str(cfg.serve_max_rq)]


@pytest.mark.parametrize("preset", PRESETS)
def test_hsextract_argv_smoke(preset):
    exp = EXPECTED[preset]
    cfg = _cfg(preset)
    s = " ".join(cmd_hsextract(cfg, Env()))
    assert "--target-layer-ids " + " ".join(str(x) for x in exp["layers"]) in s
    assert "--include-last-layer" in s
    assert f"--max-model-len {exp['seq_len']}" in s
    assert f"--tensor-parallel-size {cfg.extract_tp}" in s
    assert f"--port {cfg.extract_port}" in s
    # PP1: vLLM launch hardening present
    assert "--disable-custom-all-reduce" in s
    assert "--distributed-executor-backend mp" in s
    # stage-B offline extraction reuses the same prep/hs paths and port
    oflags = _flags(cmd_hsoffline(cfg, Env()))
    assert oflags["--preprocessed-data"] == [cfg.data_prep]
    assert oflags["--output"] == [cfg.data_hs]
    assert oflags["--endpoint"] == [f"http://127.0.0.1:{cfg.extract_port}/v1"]
    assert oflags["--request-timeout"] == ["600"]


@pytest.mark.parametrize("preset", PRESETS)
def test_data_stage_builders_smoke(preset):
    """pretokenize / prepare / regen / bench builders carry config-derived values."""
    cfg = _cfg(preset, data_regen_jsonl="/tmp/regen_in.jsonl",
               data_pretok_jsonl="/tmp/pretok.jsonl", bench_prompts="/tmp/prompts.json")
    pflags = _flags(cmd_pretokenize(cfg, Env()))
    assert pflags["--model"] == [cfg.resolved_model_path()]
    assert pflags["--input"] == ["/tmp/regen_in.jsonl"]
    assert pflags["--output"] == ["/tmp/pretok.jsonl"]
    assert "--max" not in pflags  # opt-gated branch off by default
    assert "--max" in _flags(cmd_pretokenize(
        _cfg(preset, data_regen_jsonl="/tmp/r.jsonl", data_pretok_jsonl="/tmp/p.jsonl",
             opts={"pretokenize_max": 5}), Env()))

    prflags = _flags(cmd_prepare(cfg, Env()))
    assert prflags["--seq-length"] == [str(EXPECTED[preset]["seq_len"])]
    assert "--render-endpoint" not in prflags  # PP2

    rflags = _flags(cmd_regen(cfg, Env(), endpoint="http://e:1/v1",
                              input_jsonl="/tmp/in.jsonl", output_jsonl="/tmp/out.jsonl"))
    assert rflags["--endpoint"] == ["http://e:1/v1"]
    assert rflags["--concurrency"] == [str(cfg.regen_concurrency)]

    bflags = _flags(cmd_bench(cfg, Env(), port=8010))
    assert bflags["--port"] == ["8010"]
    assert bflags["--prompts"] == ["/tmp/prompts.json"]
    assert bflags["--max-tokens"] == [str(cfg.bench_max_tokens)]


def test_train_toggles_and_passthrough():
    cfg = _cfg("qwen38-27b", fsdp=True, torch_compile=True, warm_start="/tmp/warm")
    s = " ".join(cmd_train(cfg, Env()))
    assert "--fsdp-shard" in s and "--torch-compile" in s
    assert "--from-pretrained /tmp/warm" in s


def test_train_passthrough_overrides_hardcoded_flag():
    """train.epochs/--opts must REPLACE the hardcoded --epochs, not duplicate it
    (argparse last-wins made the old duplicate silent)."""
    cfg = _cfg("qwen38-27b", opts={"train.epochs": 10, "train.scheduler_type": "linear"})
    flags = _flags(cmd_train(cfg, Env()))
    assert flags["--epochs"] == ["10"]
    # passthrough renders the engine dest name verbatim (single underscores)
    assert flags["--scheduler_type"] == ["linear"]
    cfg2 = _cfg("qwen38-27b")
    assert _flags(cmd_train(cfg2, Env()))["--epochs"] == [str(cfg2.epochs)]


@pytest.mark.parametrize("preset", PRESETS)
def test_cli_dry_run_all_stages(preset, capsys):
    """The README sanity-check promise: --stage all --dry-run must print every
    stage's argv (or an explicit skip note) and exit 0 — without any data paths."""
    cli_main(["dflash", "--preset", preset, "--stage", "all", "--dry-run"])
    out = capsys.readouterr().out
    for marker in ["=== regen", "=== pretokenize", "=== prepare", "=== hsextract",
                   "=== train (dflash)", "=== convert", "=== serve", "=== bench"]:
        assert marker in out, f"stage marker missing for {preset}: {marker}"
    # --preset path values match the preset table too (seq_len from the table)
    exp = EXPECTED[preset]
    assert f"--total-seq-len \\\n  {exp['seq_len']}" in out


def test_cli_dry_run_subset(capsys):
    """--stage takes a comma-separated subset (README example)."""
    cli_main(["dflash", "--preset", "qwen38-27b", "--stage", "train,convert,serve",
              "--dry-run"])
    out = capsys.readouterr().out
    for marker in ["=== train (dflash)", "=== convert", "=== serve"]:
        assert marker in out
    for absent in ["=== regen", "=== bench"]:
        assert absent not in out


@pytest.mark.parametrize("draft", ["dspark", "dflash2"])
def test_cli_dry_run_other_drafts(draft, capsys):
    cli_main([draft, "--preset", "qwen38-27b", "--stage", "all", "--dry-run"])
    out = capsys.readouterr().out
    assert f"=== train ({draft})" in out
