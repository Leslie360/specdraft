#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 specdraft contributors

"""Offline pretokenization: turn on-policy regen jsonl (natural language) into speculator-format rows.

The framework's prepare_data skips render for rows that already carry input_ids+loss_mask (see
preprocessing.py: pretokenized = {"input_ids","loss_mask"} <= cols).
loss_mask: 0 = prompt (system+user), 1 = the target-generated assistant response (trainable).

Usage:
  python pre_tokenize.py --model <target> --input regen.jsonl --output pretok.jsonl [--max N]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: {type(tok).__name__}, vocab={tok.vocab_size}")

    def _to_ids(x):
        """apply_chat_template(tokenize=True) may return BatchEncoding/{input_ids..}/list-of-int/Encoding/tensor —
        normalize to a list of ints. Note transformers BatchEncoding overrides the isinstance check
        (isinstance(x, dict) is False), so detection must use hasattr/get."""
        # BatchEncoding (dict subclass but isinstance dict=False)
        if hasattr(x, "get") and "input_ids" in x:
            x = x["input_ids"]
        if hasattr(x, "ids"):
            return list(x.ids)
        if hasattr(x, "tolist"):
            return [int(t) for t in x.tolist()]
        if isinstance(x, list) and x and hasattr(x[0], "ids"):
            return [t for e in x for t in e.ids]
        if isinstance(x, list) and x and isinstance(x[0], int):
            return x
        raise TypeError(f"cannot extract token ids from {type(x)}: {str(x)[:80]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skip = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            if args.max is not None and n_written >= args.max:
                break
            row = json.loads(line)
            conv = row.get("conversations") or row.get("messages")
            if not conv:
                n_skip += 1
                continue
            # keep only system/user/assistant (drop tool_call etc.)
            msgs = [m for m in conv if m.get("role") in ("system", "user", "assistant")]
            if not any(m["role"] == "assistant" for m in msgs):
                n_skip += 1
                continue
            try:
                # 27B template has no assistant_tokens_mask; the assistant turn starts at <|im_start|>
                # (our data has a single assistant turn at the end). mask = all-ones after the last <|im_start|>.
                enc = tok.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
                full_ids = _to_ids(enc)
                im_start = tok.convert_tokens_to_ids("<|im_start|>")
                starts = [i for i, t in enumerate(full_ids) if t == im_start]
                if not starts:
                    raise ValueError("no <|im_start|> found")
                last = starts[-1]  # the last one = assistant turn start
                loss_mask = [0] * last + [1] * (len(full_ids) - last)
            except Exception as e:  # noqa: BLE001
                n_skip += 1
                print(f"  skip {row.get('id', i)}: {type(e).__name__}: {str(e)[:100]}")
                continue
            if len(full_ids) != len(loss_mask):
                n_skip += 1
                print(f"  skip {row.get('id', i)}: ids/mask length mismatch {len(full_ids)} vs {len(loss_mask)}")
                continue
            out = {
                "id": row.get("id", f"row{i}"),
                "input_ids": full_ids,
                "loss_mask": loss_mask,
                "conversations": msgs,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 2000 == 0:
                print(f"  ... {n_written} written")
    print(f"done: {n_written} rows -> {args.output} (skipped {n_skip})")


if __name__ == "__main__":
    main()
