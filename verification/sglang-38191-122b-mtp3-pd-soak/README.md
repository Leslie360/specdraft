# Qwen3.5-122B MTP3 + PD (Mooncake RDMA) 1-hour soak — sgl-project/sglang#38191 verification bundle

**Setup:** 2×8 NVIDIA A800-SXM4-80GB (SM80), TP8 per side, 1P1D PD disaggregation, Mooncake 0.3.9 over 4×mlx5 IB, sgl-router, NEXTN/MTP3 (`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`), sglang main @ `03ea13a5`, Qwen3.5-122B-A10B (BF16).

**Result:** 46,826 / 46,826 requests OK over a 60-minute concurrency ladder (4 → 16 → 32), business-style prompts (`max_tokens=120`, temp=0.7); zero `AssertionError` / `OOM` / `KVTransferError` on either node; MTP3 accept length 3.06–3.41 across running batches; router `/health` 200 throughout.

## Files

- `run_122b_soak.py` — the load generator (10 business-style prompt templates, concurrency ladder)
- `soak_summary.json` — final counters (`{"ok": 46826, "fail": 0, "errors_sample": []}`)
- `soak.log` — driver-side progress log
- `launch_prefill.sh` / `launch_decode.sh` — exact server bring-up commands per node
- `prefill_excerpt.log` / `decode_excerpt.log` / `router_excerpt.log` — head/tail excerpts: startup `server_args`, RDMA topology discovery, accept-length samples, shutdown state
- `prefill_errcount.txt` / `decode_errcount.txt` — `grep -c "AssertionError|Traceback|OOM|KVTransferError"` on each full log (0 / 0)

Internal IPs, storage paths, home-directory paths, and node-identifying RDMA addresses are redacted (`10.x.x.x`, `<pfs>/...`, `<HOME>/...`, `<redacted>`); all content is otherwise verbatim. Only these redacted excerpts are distributed.
