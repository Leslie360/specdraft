#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 specdraft contributors

"""Simple accept benchmark: send N prompts to an sglang server, read spec_accept_length/rate from /metrics.

Usage: python bench_accept.py --port 8010 --prompts eval.json [--max-tokens 128] [--tag dflash1]
Output: accept_length / accept_rate / avg output tokens / tok/s
"""
import argparse
import json
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    base = f"http://127.0.0.1:{args.port}"

    # read the metrics baseline once
    def metrics():
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            return r.read().decode()
    m0 = metrics()

    total_out = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        payload = {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            total_out += len(resp["choices"][0]["message"]["content"].split())
        except Exception as e:  # noqa: BLE001
            print(f"  err {i}: {type(e).__name__}: {str(e)[:80]}")
    dt = time.time() - t0

    m1 = metrics()
    def metric(name):
        for line in m1.splitlines():
            if line.startswith(name):
                return line.split()[-1]
        return None
    # accept metrics are cumulative gauge/rate
    out = {
        "tag": args.tag,
        "prompts": len(prompts),
        "seconds": round(dt, 1),
        "avg_output_words": round(total_out / max(len(prompts), 1), 1),
        "tok_s_est": round(total_out / max(dt, 0.001), 1),
    }
    for name in ["sglang:spec_accept_length", "sglang:spec_accept_rate",
                 "sglang:spec_verify_calls_total", "sglang:spec_total_num_accepted_tokens"]:
        v = metric(name)
        if v:
            out[name.split(":")[-1]] = v
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
