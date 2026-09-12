"""Config-resolution error paths: user-input failures must exit 1 with an
'error:'-prefixed message on stderr (no traceback) — never a bare traceback,
never a silent fallback that quietly changes what runs.
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from specdraft.cli import main as cli_main  # noqa: E402
from specdraft.config import load_config  # noqa: E402
from specdraft.pipeline import run_stage  # noqa: E402
from specdraft.presets import get_preset  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("argv,needle", [
    (["dflash", "--preset", "no-such-preset", "--stage", "all", "--dry-run"],
     "unknown preset 'no-such-preset'"),
    (["dflash", "--preset", "qwen38-27b", "--stage", "bogus"],
     "unknown stage 'bogus'"),
    (["dflash", "--preset", "qwen38-27b", "--stage", "train", "--opts", "badformat"],
     "--opts must be k=v"),
    (["dflash", "--preset", "qwen38-27b", "--stage", "train", "--opts", "seq_len=abc"],
     "seq_len"),
    # task-enumerated: model_path resolution failure (neither preset nor config)
    (["dflash", "--stage", "train", "--dry-run"],
     "model_path or preset_name"),
])
def test_cli_user_errors_are_readable(argv, needle, capsys):
    """Exit code 1, 'error: ...' on stderr, no traceback."""
    with pytest.raises(SystemExit) as ei:
        cli_main(argv)
    assert ei.value.code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err and needle in captured.err
    assert "Traceback" not in captured.err and "Traceback" not in captured.out


def test_cli_missing_config_file_names_the_path(capsys, tmp_path):
    """FileNotFoundError must render via str() — 'error: 2' destroys diagnostics."""
    with pytest.raises(SystemExit) as ei:
        cli_main(["dflash", "--config", str(tmp_path / "nope.yaml"),
                  "--stage", "train", "--dry-run"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "nope.yaml" in err
    assert "Traceback" not in err


def test_cli_malformed_yaml_config_is_readable(capsys, tmp_path):
    """A typo'd YAML (the most common hand-edit mistake) must not traceback."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("seq_len: [unclosed\n  bad_indent: :", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        cli_main(["dflash", "--config", str(bad), "--stage", "train", "--dry-run"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err


def test_cli_directory_as_config_is_readable(capsys, tmp_path):
    with pytest.raises(SystemExit) as ei:
        cli_main(["dflash", "--config", str(tmp_path), "--stage", "train", "--dry-run"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err


def test_cli_missing_env_var_is_readable(monkeypatch, capsys):
    """Env() inside the CLI must get the same one-line treatment (the first
    failure every new user hits)."""
    monkeypatch.delenv("SPECDRAFT_TRAIN_PYTHON", raising=False)
    with pytest.raises(SystemExit) as ei:
        cli_main(["dflash", "--preset", "qwen38-27b", "--stage", "train", "--dry-run"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "SPECDRAFT_TRAIN_PYTHON" in err
    assert "Traceback" not in err


def test_cli_empty_stage_subset_is_rejected(capsys):
    """--stage ',' must fail loudly, not silently run the full pipeline."""
    with pytest.raises(SystemExit) as ei:
        cli_main(["dflash", "--preset", "qwen38-27b", "--stage", ",", "--dry-run"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "no stages" in err and "Traceback" not in err


def test_cli_invalid_draft_choice_rejected(capsys):
    """argparse choices: exit 2 usage error (standard), no partial execution."""
    with pytest.raises(SystemExit) as ei:
        cli_main(["mtp", "--preset", "qwen38-27b", "--stage", "train", "--dry-run"])
    assert ei.value.code == 2


def test_load_config_rejects_non_numeric_seq_len():
    """--opts seq_len=abc must raise a typed ValidationError, not store 'abc'."""
    with pytest.raises(ValidationError) as ei:
        load_config(str(REPO / "presets" / "qwen38-27b.yaml"), None, ["seq_len=abc"])
    assert "seq_len" in str(ei.value)


def test_load_config_opts_passthrough_still_works():
    """--opts routing: RunConfig fields land on the field (typed), engine
    passthrough keys (train.* / unknown) keep their values in opts."""
    cfg = load_config(str(REPO / "presets" / "qwen38-27b.yaml"), None,
                      ["train.epochs=10", "lr=6e-4", "loss_fn=ce",
                       "loss_fn={\"ce\":0.1}"])
    assert cfg.opts["train.epochs"] == 10          # engine passthrough, typed
    assert cfg.lr == 6e-4                          # RunConfig field, typed
    assert cfg.loss_fn == '{"ce":0.1}'             # RunConfig field (string JSON)


def test_run_stage_missing_data_raises_on_real_run(tmp_path):
    """Without --dry-run, a stage with unset data paths must fail loudly before
    launching anything (the dry-run path prints a skip note instead)."""
    from specdraft.config import RunConfig

    cfg = RunConfig(preset_name="qwen38-27b", data_pretok_jsonl=None, data_prep=None)
    with pytest.raises(ValueError, match="prepare requires data_pretok_jsonl"):
        run_stage("prepare", cfg, None, dry_run=False)


def test_train_real_run_requires_data_paths():
    from specdraft.config import RunConfig
    from specdraft.pipeline.train import run_train

    cfg = RunConfig(preset_name="qwen38-27b", data_prep=None, data_hs=None)
    with pytest.raises(ValueError, match="data_prep \\+ data_hs"):
        run_train(cfg, None, dry_run=False)


def test_get_preset_error_lists_available():
    with pytest.raises(KeyError) as ei:
        get_preset("qwen3-7b")
    assert "qwen3-8b" in str(ei.value)


def test_validate_gamma_mismatch_warns_but_train_run_enforces(tmp_path):
    """block/gamma mismatch stays a validation error at train time (existing
    test_validate_block_gamma covers the message; here the real-run gate)."""
    from specdraft.config import RunConfig
    from specdraft.pipeline.train import run_train

    cfg = RunConfig(preset_name="qwen38-27b", data_prep="/tmp/prep", data_hs="/tmp/hs",
                    decay_gamma=7)
    with pytest.raises(ValueError, match="decay_gamma"):
        run_train(cfg, None, dry_run=False)
