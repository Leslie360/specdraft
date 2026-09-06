"""Core module unit tests: presets / config / invoke (dry-run verifies full-pipeline argv)."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from specdraft.config import RunConfig, load_config  # noqa: E402
from specdraft.env import Env  # noqa: E402
from specdraft.invoke import cmd_train, cmd_convert, cmd_prepare  # noqa: E402
from specdraft.presets import get_preset, QWEN_PRESETS  # noqa: E402


def test_presets():
    assert set(QWEN_PRESETS) == {
        "qwen38-27b", "qwen35-122b", "qwen3-8b", "qwen35-27b",
        "qwen36-35b", "qwen38-flash-next",
    }
    p = get_preset("qwen38-27b")
    assert p.num_layers == 64 and p.hidden_size == 5120 and p.vocab_size == 248320
    assert p.target_layer_ids == (5, 19, 33, 47, 61)
    assert p.block_size == 8 and p.decay_gamma == 4
    assert p.mamba_hybrid
    # Flash-Next flagship preset adoption (production target, seq_len 65536)
    pn = get_preset("qwen38-flash-next")
    assert pn.num_layers == 48 and pn.hidden_size == 2560 and pn.vocab_size == 248320
    assert pn.target_layer_ids == (1, 7, 14, 20, 26, 32, 39, 45)
    assert pn.block_size == 8 and pn.decay_gamma == 4
    assert pn.moe and pn.mamba_hybrid and pn.seq_len == 65536


def test_load_preset_yaml():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    assert cfg.draft_vocab_size == 248320
    assert cfg.resolved_block_gamma() == (8, 4)
    assert cfg.resolved_target_layers() == ([5, 19, 33, 47, 61], 64)


def test_opts_override():
    cfg = load_config("presets/qwen38-27b.yaml", None, ["train.epochs=10", "lr=6e-4"])
    assert cfg.lr == 6e-4
    assert cfg.opts["train.epochs"] == 10


def test_train_cmd_dflash():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    cfg.draft = "dflash"
    cfg.data_prep = "/tmp/prep"
    cfg.data_hs = "/tmp/hs"
    cmd = cmd_train(cfg, Env())
    s = " ".join(cmd)
    # PP1: dflash loss + gamma
    assert "--speculator-type dflash" in s
    assert "--per-position-loss-weight dpace" in s
    assert "--dflash-decay-gamma 4" in s
    assert "--on-missing raise" in s
    # custom_all_reduce disabled (hsextract/serve commands; not in train)


def test_train_cmd_dspark():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    cfg.draft = "dspark"
    cfg.data_prep = "/tmp/prep"
    cfg.data_hs = "/tmp/hs"
    s = " ".join(cmd_train(cfg, Env()))
    assert "--markov-rank 256" in s
    assert "--enable-confidence-head" in s
    assert '{"ce":0.1,"tv":0.9}' in s


def test_train_cmd_dflash2():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    cfg.draft = "dflash2"
    cfg.data_prep = "/tmp/prep"
    cfg.data_hs = "/tmp/hs"
    s = " ".join(cmd_train(cfg, Env()))
    assert "--speculator-type dflash2" in s
    assert "--selector-rank 256" in s
    assert "--conv-kernel-size 2" in s
    assert "--sliding-window-non-causal" in s


def test_convert_cmd_dflash2_nested_config():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    cfg.draft = "dflash2"
    cmd = cmd_convert(cfg, Env())
    assert "--arch dflash2" in " ".join(cmd)
    assert "--num-target-layers 64" in " ".join(cmd)


def test_prepare_no_render():
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    cfg.data_pretok_jsonl = "/tmp/pretok.jsonl"
    cfg.data_prep = "/tmp/prep"
    s = " ".join(cmd_prepare(cfg, Env()))
    # PP2: no --render-endpoint
    assert "--render-endpoint" not in s


def test_validate_block_gamma():
    from specdraft.validate import validate_config
    cfg = load_config("presets/qwen38-27b.yaml", None, [])
    assert validate_config(cfg) == []
    cfg.decay_gamma = 7  # block8 with gamma 7 → should fail
    assert any("decay_gamma" in p for p in validate_config(cfg))
