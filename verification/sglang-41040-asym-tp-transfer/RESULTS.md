# sglang PR #41040 — synthetic byte-exact transfer validation

**Fix under test:** GLM-5.3-Flash + DFlash2 draft over Mooncake PD separation with
asymmetric P/D TP (P TP2 → D TP4): draft KV transfer used the *source* item length
as the *destination* stride, so draft writes landed at wrong addresses
(issue #41038, "fails at the tail"). The PR routes target KV through the original
path and sends draft KV head-sliced (`dst_tp_rank // ratio == src_tp_rank` selects
the serving P rank, `slice_num = dst_tp_rank % ratio` selects the head slice),
via GPU staging gather when draft pool tensors are registered, else per-token RDMA.

- base (main): `acc15c193b9d8813cb31b66cd19980fe0de9d72d`
- PR head:     `8c43def6d4f4406779bcc3d128f156e7555aebbc`

## Test method

`test_asym_dflash_transfer.py` (this directory). Real `MooncakeKVManager` from the
respective tree via `object.__new__` + hand-set fields; `conn.get_memory` monkeypatched
(`enable_unified_memory=False`); instance `_transfer_data` records every RDMA block
triple `(src, dst, len)` and replays it into a numpy bytearray memory model for
byte-exact comparison. No real RDMA. Synthetic geometry mimics GLM-5.3-Flash+DFlash2:
`page_size=64`, `head_dim=128`, bf16, 8 draft heads, 2 MLA target entries (TP
independent, item 73728 B), 2 draft layers (K+V = 4 entries). P TP2 → 4 heads/rank
(item 65536 B); D TP4 → 2 heads/rank (item 32768 B). Src pages carry per-(entry,page)
seeded random bytes; dst buffers are 0xEE sentinel. Page mapping: src
[7,3,12,1,9,0,5,2,14,6] → dst [2,0,5,3,1,7,4,6,8,9] (non-contiguous, 8+2 chunking).

## Result matrix

| scenario | tree / algo | path | checks | result |
|---|---|---|---|---|
| A control (base) | base / DFLASH | old generic path | 4 | **PASS (bug reproduced)** |
| B PR fallback | pr41040 / DFLASH | fallback per-token | 42 | **PASS** |
| C PR staging | pr41040 / DFLASH | GPU gather (Triton) | 18 | **PASS** |
| C-torch staging | pr41040 / DFLASH | GPU gather (torch fallback) | 18 | **PASS** |
| D lowercase regression | pr41040 / `dflash` | old generic path | 4 | **PASS (bug still present, = review note)** |
| E equal-TP regression | pr41040 / DFLASH | generic path | 4 | **PASS** |
| E-base reference | base / DFLASH | generic path | 4 | **PASS** |
| F TP2→TP8 (ratio=4) | pr41040 / DFLASH | fallback + staging | 45 | **PASS** |

Total: **139 checks, 0 failures.** Full logs in `logs/`, machine-readable in `json/`.

## Key quantitative evidence

### A — base repro (P TP2 → D TP4, issue #41038 "fails at the tail")
- target MLA KV: 20/20 pages byte-exact (target unaffected — expected).
- draft KV dst pages: **correct = 7266 / 2097152 bytes (0.35%)** — statistically the
  random-collision floor (~1/256 ≈ 0.39%); the transfer itself is 100% misaddressed.
- corrupted = 2089886 bytes (99.65% of the dst draft region).
- **oob = 524288 bytes exactly** = the tail chunk (dst pages 8,9) × 65536 B src-stride
  × 4 draft entries: the last chunk's writes land entirely past the 16-page dst
  region (16 × 32768 = 524288 B). This is precisely "fails at the tail": earlier
  chunks clobber in-region pages at 2× offsets, the tail chunk overruns the buffer.
  Deterministic reproduction on every run.

### D — lowercase `dflash` on the PR tree
Byte-for-byte identical numbers to A (correct=7266, corrupted=2089886, oob=524288):
the guard `speculative_algorithm == "DFLASH"` is case-sensitive, so a lowercase
value silently keeps the bug. Evidence for the PR review note.

### B — PR fallback path (no staging buffers)
For every owned pair (rank0→d0, rank0→d1, rank1→d2, rank1→d3):
- 2560 draft blocks each (4 entries × 64 tokens × 10 pages); **per-block address
  math exact: 0 bad** — src = `src_ptr + src_page*65536 + token*1024 + slice_num*512`,
  dst = `dst_ptr + dst_page*32768 + token*512`, len = 512.
- `_transfer_data` calls = 3 (target 1 + draft chunks 2, 8-page chunking confirmed).
- Full numpy replay: all 40 mapped draft pages byte-exact; 20/20 target pages
  byte-exact; zero out-of-region writes; unmapped pages keep sentinel.
- Non-owned pairs (rank0→{d2,d3}, rank1→{d0,d1}): send_kvcache returns 0, **zero
  draft blocks** (ownership gate `dst_tp_rank // ratio == src_tp_rank` verified).
- Guard rails raise RuntimeError: missing `dst_kv_item_lens`; target-stride mismatch.

### C — PR staging path (real CUDA gather, Triton kernel; C-torch = torch fallback)
For rank0→d0 and rank1→d3 with `set_draft_kv_buffer_tensors` fed real bf16 CUDA
tensors (data_ptr matching enforced by the code's own can_pack checks):
- per-thread `StagingBuffer` (1 MiB) allocated and `engine.batch_register` called.
- 40 draft blocks (4 entries × 10 pages), all sourced from the pack buffer;
  per-block addresses exact (pack offset = `Σ chunk_pages*item_len` + page_pos×item;
  dst = page-aligned).
- staging gathered bytes == expected head slices for 40/40 blocks; dst replay
  byte-exact. Triton and torch backends identical results.

### E / E-base — equal TP4 → TP4
No raise, generic path, target + draft byte-exact on both trees (behaviour
unchanged by the PR).

### F — P TP2 → D TP8 (ratio=4, dst item 16384 B, 1 head/rank)
Fallback: rank0→{d0,d3}, rank1→{d4,d7} — 2560 blocks each, address math exact,
byte-exact replay; non-owned pairs send nothing; staging variant (rank0→d3)
byte-exact through the real gather kernel.

## Import-level stubs

None at import time — both trees import cleanly under the venv (see commits.txt).
Only runtime patches: `conn.get_memory`, instance `_transfer_data`, FakeEngine,
SimpleNamespace args. The tested code paths (`send_kvcache`,
`_send_draft_kvcache_tp_slice`, `_send_kvcache_generic`, `build_transfer_entry_pairs`,
`StagingBuffer`, `gather_all_layers_to_staging`) are the real function bodies from
the respective trees.

## Pitfalls / notes

1. `run_b_pair` initially indexed draft entries with the draft-local index into the
   global `kv_data_ptrs` list — test-harness bug (index offset by `num_target`),
   caught by the harness itself; fixed, sglang code untouched.
2. Staging verification must snapshot pack-buffer bytes **at transfer time**: the
   pack buffer is reused across 8-page chunks, so post-hoc reads see the last
   chunk. The recorder resolves src bytes inside the patched `_transfer_data`,
   which mirrors real RDMA copy semantics. A spy-gather debug run confirmed the
   gather itself writes correct bytes for every chunk.
3. "correct=7266" in A/D is the random-pattern coincidence floor, not real data:
   per-byte match probability 1/256 → expected ~8192 of 2097152; observed 7266.
4. Not covered (out of scope for synthetic transfer testing): end-to-end GLM-5.3-Flash
   serving (model ~198 GB NVFP4 does not fit P TP2 on 8×A800-80GB — the reason this
   synthetic test exists), Mooncake RDMA wire behaviour, multi-PP.
