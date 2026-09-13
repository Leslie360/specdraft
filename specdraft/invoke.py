"""Per-stage command assembly (returns argv lists for --dry-run printing or subprocess execution).

Built-in fixes for known pitfalls:
- PP1  vLLM custom_all_reduce crashes on GDN → --disable-custom-all-reduce --distributed-executor-backend mp
- PP2  prepare_data requires a render-endpoint → pretokenize emits input_ids+loss_mask rows, prepare skips render
- PP4  dflash2 needs the full argv (selector/conv/sliding_window_non_causal) + full vocab
- PP6  regen is multi-instance (servers list + sharding + resume)

Script sources:
- Engine scripts (prepare/train/launch_vllm/hs offline) come from the vllm-speculators repo (SPECDRAFT_SPECULATORS_REPO)
- pretokenize / regen / convert / bench are implemented in this repo's vendor/
"""
from __future__ import annotations

from pathlib import Path

from .config import RunConfig
from .env import Env

# speculators engine script names (relative to its scripts/)
PREPARE = "prepare_data.py"
TRAIN = "train.py"
LAUNCH_VLLM = "launch_vllm.py"
HS_OFFLINE = "data_generation_offline.py"


def _vllm_args(cfg: RunConfig) -> list[str]:
    """Extra vLLM launch args (PP1 + defaults)."""
    return [
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-custom-all-reduce",       # PP1
        "--distributed-executor-backend", "mp",  # PP1
    ]


def cmd_pretokenize(cfg: RunConfig, env: Env, input_jsonl: str | None = None) -> list[str]:
    return [env.python, env.vendored("pre_tokenize.py"),
            "--model", cfg.resolved_model_path(),
            "--input", input_jsonl or cfg.data_regen_jsonl,
            "--output", cfg.data_pretok_jsonl,
            *(["--max", str(cfg.opts.get("pretokenize_max"))] if "pretokenize_max" in cfg.opts else [])]


def cmd_prepare(cfg: RunConfig, env: Env) -> list[str]:
    """prepare_data: never pass --render-endpoint (PP2: pretokenize rows carry input_ids+loss_mask)."""
    return [env.python, env.script(PREPARE),
            "--model", cfg.resolved_model_path(),
            "--data", cfg.data_pretok_jsonl,
            "--output", cfg.data_prep,
            "--seq-length", str(cfg.seq_len),
            "--token-freq-path", cfg.data_token_freq or f"{cfg.data_prep}/token_freq.pt",
            "--trust-remote-code",
            "--overwrite"]


def cmd_hsextract(cfg: RunConfig, env: Env, *, gpus: str | None = None) -> list[str]:
    tl, _ = cfg.resolved_target_layers()
    return [env.python, env.script(LAUNCH_VLLM), cfg.resolved_model_path(),
            "--target-layer-ids", *map(str, tl),
            "--include-last-layer",
            "--hidden-states-path", "/tmp/hs_qdt",
            "--",
            "--port", str(cfg.extract_port),
            "--tensor-parallel-size", str(cfg.extract_tp),
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", str(cfg.seq_len),
            *_vllm_args(cfg)]


def cmd_hsoffline(cfg: RunConfig, env: Env) -> list[str]:
    # engine semantics: --max-samples 0 means "process zero samples"; omitting it
    # means unlimited. Only pass it through when the user sets hs_max_samples.
    max_samples = cfg.opts.get("hs_max_samples")
    return [env.python, env.script(HS_OFFLINE),
            "--preprocessed-data", cfg.data_prep,
            "--output", cfg.data_hs,
            *(["--max-samples", str(max_samples)] if max_samples else []),
            "--endpoint", f"http://127.0.0.1:{cfg.extract_port}/v1",
            "--concurrency", str(cfg.opts.get("hs_concurrency", 32)),
            "--request-timeout", "600"]


def _loss_flags(cfg: RunConfig) -> list[str]:
    """Loss/head flags per draft type."""
    if cfg.draft == "dflash":
        return ["--loss-fn", cfg.loss_fn or "ce",
                "--per-position-loss-weight", "dpace",
                "--dflash-decay-gamma", str(cfg.resolved_block_gamma()[1])]
    if cfg.draft == "dspark":
        return ["--loss-fn", cfg.loss_fn or '{"ce":0.1,"tv":0.9}',
                "--markov-rank", "256", "--markov-head-type", "vanilla",
                "--enable-confidence-head", "--confidence-head-with-markov",
                "--confidence-head-alpha", "1.0",
                "--per-position-loss-weight", "fixed-exp-decay",
                "--dflash-decay-gamma", str(cfg.resolved_block_gamma()[1])]
    if cfg.draft == "dflash2":
        # PP4: full dflash2 argv + full vocab
        return ["--loss-fn", cfg.loss_fn or "kl_div",
                "--selector-rank", "256", "--selector-top-k", "16",
                "--conv-kernel-size", "2", "--conv-group-size", "16",
                "--sliding-window-non-causal",
                "--per-position-loss-weight", "fixed-exp-decay",
                "--dflash-decay-gamma", str(cfg.resolved_block_gamma()[1])]
    raise ValueError(f"unknown draft: {cfg.draft}")


def cmd_train(cfg: RunConfig, env: Env) -> list[str]:
    tl, ntl = cfg.resolved_target_layers()
    vocab = cfg.resolved_vocab()
    # engine passthrough opts are the highest-precedence override: a train.* key
    # mapping to an already-emitted flag (--epochs/--lr) replaces it instead of
    # emitting a contradictory duplicate
    overrides = {k.removeprefix("train."): v for k, v in cfg.opts.items()
                 if k.startswith("train.")}
    epochs = overrides.pop("epochs", cfg.epochs)
    lr = overrides.pop("lr", cfg.lr)
    # data paths default to an explicit "<unset>" placeholder (dry-run must always
    # assemble); the train stage hard-checks them on a real run
    cmd = [env.python, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", str(cfg.nproc),
           env.script(TRAIN),
           "--verifier-name-or-path", cfg.resolved_model_path(),
           "--data-path", cfg.data_prep or "<unset>",
           "--hidden-states-path", cfg.data_hs or "<unset>", "--on-missing", "raise",
           "--save-path", cfg.resolved_ckpt(),
           "--draft-vocab-size", str(vocab),
           "--total-seq-len", str(cfg.seq_len),
           "--speculator-type", cfg.draft,
           "--block-size", str(cfg.resolved_block_gamma()[0]),
           "--max-anchors", str(cfg.max_anchors),
           "--num-layers", str(cfg.num_layers),
           "--target-layer-ids", *map(str, tl),
           "--lr", str(lr),
           "--epochs", str(epochs),
           *_loss_flags(cfg),
           "--trust-remote-code"]
    if cfg.fsdp:
        cmd.append("--fsdp-shard")
    if cfg.torch_compile:
        cmd.append("--torch-compile")
    if cfg.warm_start:
        cmd.append("--from-pretrained")
        cmd.append(cfg.warm_start)
    # remaining engine passthrough opts → append --k v (engine CLI renders dests
    # with underscores as hyphens: --max_steps -> --max-steps)
    for dest, v in overrides.items():
        cmd += [f"--{dest.replace('_', '-')}", str(v)]
    return cmd


def _convert_source_dir(cfg: RunConfig) -> str:
    """The engine saves checkpoints as ckpt/<name>/config.json — the convert step
    must point at the checkpoint subdirectory, not the ckpt parent. Prefer
    checkpoint_best (the engine's tracked best), falling back to the newest
    epoch*_end dir when checkpoint_best is absent."""
    ckpt_root = Path(cfg.resolved_ckpt())
    best = ckpt_root / "checkpoint_best"
    if best.is_dir() and (best / "config.json").is_file():
        return str(best)
    ends = sorted(ckpt_root.glob("epoch*_end"))
    if ends:
        return str(ends[-1])
    return str(ckpt_root)


def cmd_convert(cfg: RunConfig, env: Env) -> list[str]:
    return [env.python, env.vendored("convert_servable.py"),
            _convert_source_dir(cfg), cfg.resolved_servable(),
            "--arch", cfg.draft,
            "--num-target-layers", str(cfg.resolved_target_layers()[1])]


def cmd_serve(cfg: RunConfig, env: Env, *, port: int, gpus: str, tp: int,
              draft_model: str | None = None) -> list[str]:
    algo = "NEXTN" if cfg.draft == "mtp" else "DFLASH"
    draft = draft_model or cfg.spec_draft_model or cfg.resolved_servable()
    cmd = [env.sglang_python, "-m", "sglang.launch_server",
           "--model-path", cfg.resolved_model_path(),
           "--served-model-name", cfg.resolved_model_path().split("/")[-1],
           "--tp", str(tp),
           "--context-length", str(cfg.serve_ctx),
           "--chunked-prefill-size", str(cfg.serve_ctx),
           "--mem-fraction-static", str(cfg.serve_mem_frac),
           "--enable-p2p-check", "--enable-metrics",
           "--max-running-requests", str(cfg.serve_max_rq),
           "--port", str(port),
           "--speculative-algorithm", algo,
           "--speculative-draft-model-path", draft,
           # DFLASH proposes one token per block slot, so num-draft-tokens must
           # equal the draft's block_size (serve with a mismatched count wastes
           # drafts at best and distorts acceptance at worst).
           "--speculative-num-draft-tokens", str(cfg.resolved_block_gamma()[0])]
    return cmd


def cmd_bench(cfg: RunConfig, env: Env, *, port: int) -> list[str]:
    return [env.python, env.vendored("bench_accept.py"),
            "--port", str(port),
            "--prompts", cfg.bench_prompts or "",
            "--max-tokens", str(cfg.bench_max_tokens)]


def cmd_regen(cfg: RunConfig, env: Env, *, endpoint: str, input_jsonl: str, output_jsonl: str) -> list[str]:
    return [env.python, env.vendored("regen_responses.py"),
            "--endpoint", endpoint,
            # the served model name must match the server's model id; when a
            # server is launched without --served-model-name, vLLM uses the
            # model *path* as the id, so pass the resolved path through
            "--model", cfg.resolved_model_path(),
            "--input", input_jsonl,
            "--output", output_jsonl,
            "--concurrency", str(cfg.regen_concurrency)]
