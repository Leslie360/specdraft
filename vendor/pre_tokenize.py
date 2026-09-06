#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 specdraft contributors

"""离线预 tokenize：把 on-policy regen jsonl（自然语言）转成 speculator-format 行。

框架的 prepare_data 对含 input_ids+loss_mask 的行自动跳过 render（见
preprocessing.py: pretokenized = {"input_ids","loss_mask"} <= cols）。
loss_mask: 0 = prompt（system+user），1 = 目标生成的 assistant 响应（可训）。

用法:
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
        """apply_chat_template(tokenize=True) 可能返回 BatchEncoding/{input_ids..}/int列表/Encoding/张量，统一成 int 列表。
        注意 transformers BatchEncoding 重写了 isinstance 检查（isinstance(x, dict) 为 False），
        必须用 hasattr/get 检测。"""
        # BatchEncoding（dict 子类但 isinstance dict=False）
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
        raise TypeError(f"无法从 {type(x)} 提取 token ids: {str(x)[:80]}")

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
            # 只留 system/user/assistant（丢弃 tool_call 等，open-perfectblend 无）
            msgs = [m for m in conv if m.get("role") in ("system", "user", "assistant")]
            if not any(m["role"] == "assistant" for m in msgs):
                n_skip += 1
                continue
            try:
                # 27B 模板不支持 assistant_tokens_mask；assistant 轮以 <|im_start|> 开头（我们的数据
                # 是单个 assistant 轮在末尾）。mask = 最后一个 <|im_start|> 之后全 1（目标生成区）。
                enc = tok.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
                full_ids = _to_ids(enc)
                im_start = tok.convert_tokens_to_ids("<|im_start|>")
                starts = [i for i, t in enumerate(full_ids) if t == im_start]
                if not starts:
                    raise ValueError("no <|im_start|> found")
                last = starts[-1]  # 最后一个 = assistant 轮起始
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
