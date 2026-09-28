#!/usr/bin/env python3
"""
Synthetic byte-exact transfer validation for sglang PR #41040
(fix: transfer DFlash draft KV across asymmetric TP via Mooncake PD).

Method
------
* The real `MooncakeKVManager` from either the base (main) tree or the PR
  tree is instantiated with ``object.__new__`` (no __init__, no engine, no
  RDMA).  Only the attributes touched by the code under test are set.
* Module-level ``get_memory`` inside ``conn.py`` is monkeypatched to return
  ``enable_unified_memory=False`` (so the envelope validator early-returns).
* The instance method ``_transfer_data`` is replaced by a recorder that
  captures every (src, dst, len) RDMA block triple and replays them into a
  numpy-backed virtual memory model for byte-exact checks.
* Source pages are filled with per-(entry, page) seeded random bytes;
  destination buffers are filled with a 0xEE sentinel.

Scenarios
---------
A      base tree, P TP2 -> D TP4 : reproduce the bug (src stride used as dst
       stride for draft KV); quantify misplaced / out-of-region bytes.
B      PR tree, fallback path (no _draft_kv_buffers), P TP2 -> D TP4:
       exact per-block address math + full byte-exact replay for every
       (engine_rank, dst_tp_rank) ownership combination, plus guard rails.
C      PR tree, GPU staging path (set_draft_kv_buffer_tensors with real CUDA
       tensors): verify gather output in the staging buffer is byte-exact and
       block target addresses are correct.  Default Triton kernel and, via
       module reload, the torch fallback.
D      PR tree but speculative_algorithm="dflash" (lowercase): branch must
       NOT trigger; behaviour must equal scenario A (review-note evidence).
E      PR tree, equal TP4 -> TP4: generic path, no raise, byte-exact content.
E-base base tree, equal TP4 -> TP4: identical behaviour reference.
F      PR tree, P TP2 -> D TP8 (ratio=4): fallback + staging byte-exact.
"""

import argparse
import concurrent.futures
import importlib
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic model geometry, mimicking GLM-5.3-Flash + DFlash2 draft
# ---------------------------------------------------------------------------
PAGE_SIZE = 64
HEAD_DIM = 128
DTYPE_BYTES = 2  # bf16
DRAFT_TOTAL_HEADS = 8
DRAFT_LAYERS = 2
TARGET_ENTRIES = 2  # MLA latent entries; layout identical on every TP rank
TARGET_ITEM_LEN = PAGE_SIZE * 576 * DTYPE_BYTES  # 64 * 576 * 2 = 73728
POOL_PAGES = 16

NUM_DRAFT = DRAFT_LAYERS * 2  # K layers then V layers, appended after target
SRC_LAYER_IDS = [0, 1] + [100, 101, 100, 101]
DST_LAYER_IDS = list(SRC_LAYER_IDS)

SRC_PAGES = np.array([7, 3, 12, 1, 9, 0, 5, 2, 14, 6], dtype=np.int32)
DST_PAGES = np.array([2, 0, 5, 3, 1, 7, 4, 6, 8, 9], dtype=np.int32)
SRC_LIST = [int(x) for x in SRC_PAGES]
DST_LIST = [int(x) for x in DST_PAGES]

EXECUTOR = concurrent.futures.ThreadPoolExecutor(4)
CHECKS = []  # (scenario, name, ok, detail)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(scenario, name, ok, detail=""):
    CHECKS.append((scenario, name, bool(ok), detail))
    log(f"{'PASS' if ok else 'FAIL'} [{scenario}] {name} :: {detail}")
    return bool(ok)


# ---------------------------------------------------------------------------
# Virtual memory model: one bytearray, regions are arbitrary int addresses
# ---------------------------------------------------------------------------
class VMem:
    def __init__(self, base=0x100000, gap=1 << 20):
        self.base = base
        self.cursor = base
        self.gap = gap
        self.mem = bytearray()
        self.regions = {}
        self.oob_bytes = 0
        self.oob_blocks = 0

    def alloc(self, name, size):
        start = self.cursor
        self.regions[name] = (start, size)
        need = start - self.base + size
        if need > len(self.mem):
            self.mem.extend(b"\x00" * (need - len(self.mem)))
        self.cursor = start + size + self.gap
        return start

    def fill(self, ptr, size, byte):
        off = ptr - self.base
        self.mem[off : off + size] = bytes([byte]) * size

    def find(self, ptr):
        for name, (s, sz) in self.regions.items():
            if s <= ptr < s + sz:
                return name, s, sz
        return None, 0, 0

    def read(self, ptr, n):
        off = ptr - self.base
        return bytes(self.mem[off : off + n])

    def write(self, ptr, data):
        name, s, sz = self.find(ptr)
        if name is None or ptr + len(data) > s + sz:
            self.oob_bytes += len(data)
            self.oob_blocks += 1
            return False
        off = ptr - self.base
        self.mem[off : off + len(data)] = data
        return True


def pattern_bytes(entry_idx, page_id, item_len):
    rng = np.random.default_rng((0x41040 << 24) ^ (entry_idx << 12) ^ page_id)
    return rng.integers(0, 256, size=item_len, dtype=np.uint8).tobytes()


class Layout:
    """Builds the src/dst buffer layouts for a given (src_tp, dst_tp) pair."""

    def __init__(self, src_tp, dst_tp):
        self.src_tp, self.dst_tp = src_tp, dst_tp
        self.src_heads = DRAFT_TOTAL_HEADS // src_tp
        self.dst_heads = DRAFT_TOTAL_HEADS // dst_tp
        self.src_draft_item = PAGE_SIZE * self.src_heads * HEAD_DIM * DTYPE_BYTES
        self.dst_draft_item = PAGE_SIZE * self.dst_heads * HEAD_DIM * DTYPE_BYTES
        self.ratio = dst_tp // src_tp if dst_tp > src_tp else 1

        self.vm_src, self.vm_dst = VMem(), VMem()
        self.src_ptrs, self.src_lens = [], []
        self.dst_ptrs, self.dst_lens = [], []
        self.content = {}  # ("t"|"d", entry, page) -> bytes

        for e in range(TARGET_ENTRIES):
            p = self.vm_src.alloc(f"src_t{e}", POOL_PAGES * TARGET_ITEM_LEN)
            self.src_ptrs.append(p)
            self.src_lens.append(TARGET_ITEM_LEN)
            for pg in range(POOL_PAGES):
                b = pattern_bytes(e, pg, TARGET_ITEM_LEN)
                self.content[("t", e, pg)] = b
                self.vm_src.write(p + pg * TARGET_ITEM_LEN, b)
        for e in range(NUM_DRAFT):
            p = self.vm_src.alloc(f"src_d{e}", POOL_PAGES * self.src_draft_item)
            self.src_ptrs.append(p)
            self.src_lens.append(self.src_draft_item)
            for pg in range(POOL_PAGES):
                b = pattern_bytes(100 + e, pg, self.src_draft_item)
                self.content[("d", e, pg)] = b
                self.vm_src.write(p + pg * self.src_draft_item, b)

        for e in range(TARGET_ENTRIES):
            p = self.vm_dst.alloc(f"dst_t{e}", POOL_PAGES * TARGET_ITEM_LEN)
            self.dst_ptrs.append(p)
            self.dst_lens.append(TARGET_ITEM_LEN)
            self.vm_dst.fill(p, POOL_PAGES * TARGET_ITEM_LEN, 0xEE)
        for e in range(NUM_DRAFT):
            p = self.vm_dst.alloc(f"dst_d{e}", POOL_PAGES * self.dst_draft_item)
            self.dst_ptrs.append(p)
            self.dst_lens.append(self.dst_draft_item)
            self.vm_dst.fill(p, POOL_PAGES * self.dst_draft_item, 0xEE)

    def expected_target_page(self, entry, page):
        return self.content[("t", entry, page)]

    def expected_draft_page(self, src_entry, src_page, src_tp_rank, dst_tp_rank):
        """Bytes a dst page must hold: the dst_tp_rank's global-head slice of
        the owning prefill rank's local shard, for every token of the page."""
        dst_tok = self.dst_heads * HEAD_DIM * DTYPE_BYTES
        src_tok = self.src_heads * HEAD_DIM * DTYPE_BYTES
        g0 = dst_tp_rank * self.dst_heads
        local = g0 - src_tp_rank * self.src_heads
        assert 0 <= local and local + self.dst_heads <= self.src_heads
        page = self.content[("d", src_entry, src_page)]
        out = bytearray()
        for t in range(PAGE_SIZE):
            off = t * src_tok + local * HEAD_DIM * DTYPE_BYTES
            out += page[off : off + dst_tok]
        return bytes(out)


# ---------------------------------------------------------------------------
# Manager construction (object.__new__, no __init__) and patching
# ---------------------------------------------------------------------------
class FakeEngine:
    def __init__(self):
        self.registered = []
        self.deregistered = []

    def batch_register(self, ptrs, lens=None):
        self.registered.append((list(ptrs), list(lens) if lens is not None else None))
        return 0

    def batch_deregister(self, ptrs):
        self.deregistered.append(list(ptrs))
        return 0

    def batch_transfer_sync(self, sid, src, dst, lens):
        return 0


def make_manager(conn, spec_algo, src_tp, engine_rank, lay):
    mgr = object.__new__(conn.MooncakeKVManager)
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=list(lay.src_ptrs),
        kv_data_lens=[POOL_PAGES * l for l in lay.src_lens],
        kv_item_lens=list(lay.src_lens),
        kv_layer_ids=list(SRC_LAYER_IDS),
        num_draft_entries=NUM_DRAFT,
        page_size=PAGE_SIZE,
        engine_rank=engine_rank,
        gpu_id=0,
        prefill_start_layer=0,
        prefill_end_layer=None,
        mla_compression_ratios=None,
        system_dp_rank=0,
    )
    mgr.server_args = SimpleNamespace(speculative_algorithm=spec_algo)
    mgr.attn_tp_size = src_tp
    mgr.pp_size = 1
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr.enable_custom_mem_pool = False
    mgr.custom_mem_pool_type = None
    mgr.max_transfer_batch_indices = 0
    mgr.enable_deferred_decode_kv_release = False
    mgr.disaggregation_mode = None
    mgr.enable_staging = False
    mgr._draft_kv_buffers = None
    mgr._draft_pack_buffers = {}
    mgr.engine = FakeEngine()
    return mgr


def patch_conn_module(conn):
    conn.get_memory = lambda: SimpleNamespace(enable_unified_memory=False)


class Recorder:
    """Replacement for _transfer_data: records blocks, replays into VMem.

    Src bytes are resolved at record time (i.e. at RDMA time) so that a reused
    per-thread staging buffer cannot contaminate later chunks."""

    def __init__(self, vm_src=None, pack_resolver=None):
        self.blocks = []  # (src, dst, len, src_bytes)
        self.calls = 0
        self.vm_src = vm_src
        self.pack_resolver = pack_resolver
        self.pack_reads = 0

    def __call__(self, session_id, transfer_blocks):
        self.calls += 1
        for s, d, l in transfer_blocks:
            s, d, l = int(s), int(d), int(l)
            src_bytes = None
            if self.vm_src is not None:
                name, st, sz = self.vm_src.find(s)
                if name is not None and s + l <= st + sz:
                    src_bytes = self.vm_src.read(s, l)
            if src_bytes is None and self.pack_resolver is not None:
                src_bytes = self.pack_resolver(s, l)
                if src_bytes is not None:
                    self.pack_reads += 1
            if src_bytes is None:
                raise AssertionError(f"unresolvable src {s:#x} len {l}")
            self.blocks.append((s, d, l, src_bytes))
        return 0

    def replay_into(self, vm_dst):
        for s, d, l, src_bytes in self.blocks:
            vm_dst.write(d, src_bytes)
        return vm_dst.oob_bytes


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify_target_content(label, lay):
    ok, bad = 0, 0
    for e in range(TARGET_ENTRIES):
        for pos, d in enumerate(DST_LIST):
            got = lay.vm_dst.read(
                lay.dst_ptrs[e] + d * TARGET_ITEM_LEN, TARGET_ITEM_LEN
            )
            if got == lay.expected_target_page(e, SRC_LIST[pos]):
                ok += 1
            else:
                bad += 1
    check(label, "target KV byte-exact", bad == 0,
          f"pages_ok={ok} pages_bad={bad}")
    return ok, bad


def draft_region_stats(label, lay):
    """Final content of dst draft regions vs correct slice semantics."""
    dst_tok = lay.dst_heads * HEAD_DIM * DTYPE_BYTES
    total = NUM_DRAFT * POOL_PAGES * lay.dst_draft_item
    correct = corrupted = 0
    mapped = {d: s for d, s in zip(DST_LIST, SRC_LIST)}
    for j in range(NUM_DRAFT):
        base = lay.dst_ptrs[TARGET_ENTRIES + j]
        for pg in range(POOL_PAGES):
            got = lay.vm_dst.read(base + pg * lay.dst_draft_item,
                                  lay.dst_draft_item)
            if pg in mapped:
                want = lay.expected_draft_page(j, mapped[pg], 0, 0)
                c = sum(a == b for a, b in zip(got, want))
                correct += c
                corrupted += lay.dst_draft_item - c
            else:
                if any(b != 0xEE for b in got):
                    corrupted += lay.dst_draft_item
    return dict(total=total, correct=correct, corrupted=corrupted)


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------
def load_conn(tree):
    sys.path.insert(0, os.path.join(tree, "python"))
    import sglang.srt.disaggregation.mooncake.conn as conn

    patch_conn_module(conn)
    log(f"conn module: {conn.__file__}")
    return conn


def entry_pairs(conn):
    from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

    return build_transfer_entry_pairs(
        SRC_LAYER_IDS[TARGET_ENTRIES:], DST_LAYER_IDS[TARGET_ENTRIES:],
        NUM_DRAFT, NUM_DRAFT, allow_positional_fallback=True,
    )


def scenario_a_or_d(conn, label, spec_algo):
    """Base-tree repro (A) or PR-tree lowercase repro (D)."""
    src_tp, dst_tp = 2, 4
    lay = Layout(src_tp, dst_tp)
    mgr = make_manager(conn, spec_algo, src_tp, engine_rank=0, lay=lay)
    rec = Recorder(vm_src=lay.vm_src)
    mgr._transfer_data = rec
    mgr.send_kvcache(
        "s1", SRC_PAGES, lay.dst_ptrs, DST_PAGES, EXECUTOR,
        dst_layer_ids=DST_LAYER_IDS, dst_kv_item_len=TARGET_ITEM_LEN,
        dst_attn_tp_size=dst_tp,
    )
    log(f"recorded {len(rec.blocks)} blocks in {rec.calls} call(s)")
    oob = rec.replay_into(lay.vm_dst)

    verify_target_content(label, lay)
    st = draft_region_stats(label, lay)
    log(f"draft bytes: total={st['total']} correct={st['correct']} "
        f"corrupted={st['corrupted']} oob={oob}")
    # Deterministic repro: the tail chunk (dst pages 8,9 -> offsets 8*65536,
    # 9*65536) overruns the 16-page dst region entirely: 2 pages * 65536 B
    # * 4 draft entries land out of region.
    want_oob = 2 * lay.src_draft_item * NUM_DRAFT
    # Random-byte collision floor is ~1/256 = 0.39%; the observed "correct"
    # count must stay in the coincidence range, i.e. the transfer itself is
    # 100% misaddressed.
    correct_ratio = st["correct"] / st["total"]
    check(label, "draft dst pages byte-exact", correct_ratio < 0.01,
          f"correct={st['correct']}/{st['total']} ratio={correct_ratio:.4f}")
    check(label, "tail-chunk writes overrun dst buffer (fails at the tail)",
          oob == want_oob, f"oob={oob} expected={want_oob}")
    check(label, "every in-region draft byte wrong or clobbered",
          st["corrupted"] >= st["total"] * 0.99,
          f"corrupted={st['corrupted']}/{st['total']}")


def run_b_pair(conn, label, src_tp, dst_tp, engine_rank, dst_tp_rank,
               use_staging=False, staging_torch=False):
    """Core check for one (engine_rank, dst_tp_rank) combination."""
    lay = Layout(src_tp, dst_tp)
    mgr = make_manager(conn, "DFLASH", src_tp, engine_rank=engine_rank, lay=lay)
    pairs = entry_pairs(conn)
    assert pairs == [(i, i) for i in range(NUM_DRAFT)], pairs

    ratio = dst_tp // src_tp
    src_tp_rank = engine_rank % src_tp
    owned = (dst_tp_rank // ratio) == src_tp_rank
    slice_num = dst_tp_rank % ratio
    log(f"{label}: src_tp={src_tp} dst_tp={dst_tp} engine_rank={engine_rank} "
        f"dst_tp_rank={dst_tp_rank} owned={owned} slice_num={slice_num}")

    pack_resolver = None
    if use_staging:
        import torch

        k_bufs, v_bufs, draft_ptrs = [], [], []
        for e in range(NUM_DRAFT):
            data = bytearray()
            for pg in range(POOL_PAGES):
                data += lay.content[("d", e, pg)]
            cpu = torch.frombuffer(bytes(data), dtype=torch.uint8).clone()
            t = cpu.cuda().view(torch.bfloat16).reshape(
                POOL_PAGES * PAGE_SIZE, lay.src_heads, HEAD_DIM
            )
            (k_bufs if e < NUM_DRAFT // 2 else v_bufs).append(t)
            draft_ptrs.append(t.data_ptr())
        lay.src_ptrs[TARGET_ENTRIES:] = draft_ptrs
        mgr.kv_args.kv_data_ptrs = list(lay.src_ptrs)
        mgr._draft_kv_buffers = (k_bufs, v_bufs)
        if staging_torch:
            os.environ["SGLANG_STAGING_USE_TORCH"] = "1"
            import sglang.srt.disaggregation.common.staging_buffer as sb

            importlib.reload(sb)
            log(f"staging gather backend reloaded: _USE_TRITON_STAGING="
                f"{sb._USE_TRITON_STAGING}")

        def pack_resolver(ptr, n):  # noqa: F811
            for buf in mgr._draft_pack_buffers.values():
                p = buf.get_ptr()
                if p <= ptr and ptr + n <= p + buf.get_size():
                    return buf.buffer[ptr - p : ptr - p + n].cpu().numpy().tobytes()
            return None

    rec = Recorder(vm_src=lay.vm_src, pack_resolver=pack_resolver)
    mgr._transfer_data = rec
    ret = mgr.send_kvcache(
        "s1", SRC_PAGES, lay.dst_ptrs, DST_PAGES, EXECUTOR,
        dst_layer_ids=DST_LAYER_IDS, dst_kv_item_len=TARGET_ITEM_LEN,
        dst_attn_tp_size=dst_tp,
        dst_kv_item_lens=list(lay.dst_lens), dst_tp_rank=dst_tp_rank,
    )
    check(label, "send_kvcache returns 0", ret == 0, f"ret={ret}")

    draft_blocks = []
    for s, d, l, _ in rec.blocks:
        name, _, _ = lay.vm_dst.find(d)
        if name is not None and name.startswith("dst_d"):
            draft_blocks.append((s, d, l))

    if not owned:
        check(label, "non-owned pair sends zero draft blocks",
              len(draft_blocks) == 0, f"draft_blocks={len(draft_blocks)}")
        return lay, rec, mgr

    # ---- per-block address verification ----------------------------------
    src_tok = lay.src_heads * HEAD_DIM * DTYPE_BYTES
    dst_tok = lay.dst_heads * HEAD_DIM * DTYPE_BYTES
    n_addr_bad = 0
    for s, d, l in draft_blocks:
        name, _, _ = lay.vm_dst.find(d)
        j = int(name[5:])  # draft-local entry index (dst_d{j})
        i = pairs[j][0]
        gj = TARGET_ENTRIES + j   # global index into lay.dst_ptrs
        gi = TARGET_ENTRIES + i   # global index into lay.src_ptrs
        rel = d - lay.dst_ptrs[gj]
        dst_page, rem = divmod(rel, lay.dst_draft_item)
        pos = DST_LIST.index(dst_page)
        src_page = SRC_LIST[pos]
        if use_staging:
            page_pos = pos % 8
            chunk_start = pos - page_pos
            chunk_pages = min(8, len(SRC_LIST) - chunk_start)
            off = 0
            for jj in range(j):
                off += chunk_pages * lay.dst_draft_item
            ok_addr = (rem == 0 and l == lay.dst_draft_item)
            if len(mgr._draft_pack_buffers) == 1:
                p_base = next(iter(mgr._draft_pack_buffers.values())).get_ptr()
                ok_addr = ok_addr and s == p_base + off + page_pos * lay.dst_draft_item
            else:
                ok_addr = False
        else:
            tok, off2 = divmod(rem, dst_tok)
            ok_addr = (l == dst_tok and off2 == 0 and 0 <= tok < PAGE_SIZE)
            src_rel = s - lay.src_ptrs[gi]
            if src_rel < 0:
                ok_addr = False
            else:
                s_pg, s_rem = divmod(src_rel, lay.src_draft_item)
                s_tok, head_off = divmod(s_rem, src_tok)
                ok_addr = (ok_addr and s_pg == src_page and s_tok == tok
                           and head_off == slice_num * dst_tok)
        if not ok_addr:
            n_addr_bad += 1
            if n_addr_bad <= 3:
                log(f"  BAD BLOCK s={s:#x} d={d:#x} l={l}")
    check(label, "per-block src/dst address math exact", n_addr_bad == 0,
          f"draft_blocks={len(draft_blocks)} bad={n_addr_bad}")

    if not use_staging:
        want_blocks = NUM_DRAFT * PAGE_SIZE * len(SRC_LIST)
        check(label, "fallback block count (per-token blocks)",
              len(draft_blocks) == want_blocks,
              f"got={len(draft_blocks)} want={want_blocks}")
        want_calls = 1 + (len(SRC_LIST) + 7) // 8
        check(label, "transfer_data call count (target + 8-page chunks)",
              rec.calls == want_calls, f"calls={rec.calls} want={want_calls}")

    # ---- byte-exact content verification ---------------------------------
    oob = rec.replay_into(lay.vm_dst)
    check(label, "no out-of-region writes", oob == 0, f"oob={oob}")
    verify_target_content(label, lay)

    n_bad_pages = 0
    for j in range(NUM_DRAFT):
        i = pairs[j][0]
        gj = TARGET_ENTRIES + j
        for pos, d in enumerate(DST_LIST):
            got = lay.vm_dst.read(
                lay.dst_ptrs[gj] + d * lay.dst_draft_item, lay.dst_draft_item)
            if got != lay.expected_draft_page(i, SRC_LIST[pos],
                                              src_tp_rank, dst_tp_rank):
                n_bad_pages += 1
    check(label, "draft dst pages byte-exact (all mapped pages)",
          n_bad_pages == 0, f"bad={n_bad_pages}/{NUM_DRAFT * len(DST_LIST)}")

    mapped = set(DST_LIST)
    n_dirty = 0
    for j in range(NUM_DRAFT):
        gj = TARGET_ENTRIES + j
        for pg in range(POOL_PAGES):
            if pg not in mapped:
                got = lay.vm_dst.read(
                    lay.dst_ptrs[gj] + pg * lay.dst_draft_item,
                    lay.dst_draft_item)
                if any(b != 0xEE for b in got):
                    n_dirty += 1
    check(label, "unmapped dst pages untouched (sentinel)", n_dirty == 0,
          f"dirty={n_dirty}")

    if use_staging:
        check(label, "staging pack buffer allocated & registered",
              len(mgr._draft_pack_buffers) > 0 and len(mgr.engine.registered) > 0,
              f"pack_buffers={len(mgr._draft_pack_buffers)} "
              f"engine_regs={len(mgr.engine.registered)}")
        check(label, "all draft blocks sourced from staging buffer",
              rec.pack_reads == len(draft_blocks),
              f"pack_reads={rec.pack_reads} draft_blocks={len(draft_blocks)}")
        n_stg_bad = 0
        for s, d, l, src_bytes in rec.blocks:
            name, _, _ = lay.vm_dst.find(d)
            if name is None or not name.startswith("dst_d"):
                continue
            j = int(name[5:])
            i = pairs[j][0]
            dst_page = (d - lay.dst_ptrs[TARGET_ENTRIES + j]) // lay.dst_draft_item
            pos = DST_LIST.index(dst_page)
            want = lay.expected_draft_page(i, SRC_LIST[pos],
                                           src_tp_rank, dst_tp_rank)
            if src_bytes != want:
                n_stg_bad += 1
        check(label, "staging gathered bytes == expected head slices",
              n_stg_bad == 0, f"bad={n_stg_bad}/{len(draft_blocks)}")
    return lay, rec, mgr


def scenario_b(conn):
    src_tp, dst_tp = 2, 4
    for engine_rank, dst_tp_rank in [(0, 0), (0, 1), (1, 2), (1, 3)]:
        run_b_pair(conn, f"B rank{engine_rank}->d{dst_tp_rank}",
                   src_tp, dst_tp, engine_rank, dst_tp_rank)
    for engine_rank, dst_tp_rank in [(0, 2), (0, 3), (1, 0), (1, 1)]:
        run_b_pair(conn, f"Bneg-owner rank{engine_rank}->d{dst_tp_rank}",
                   src_tp, dst_tp, engine_rank, dst_tp_rank)
    # guard rails
    lay = Layout(src_tp, dst_tp)
    mgr = make_manager(conn, "DFLASH", src_tp, engine_rank=0, lay=lay)
    rec = Recorder(vm_src=lay.vm_src)
    mgr._transfer_data = rec
    try:
        mgr.send_kvcache("s1", SRC_PAGES, lay.dst_ptrs, DST_PAGES, EXECUTOR,
                         dst_attn_tp_size=dst_tp, dst_tp_rank=0)
        check("Bneg-guard", "missing dst_kv_item_lens raises RuntimeError",
              False, "no exception raised")
    except RuntimeError as e:
        check("Bneg-guard", "missing dst_kv_item_lens raises RuntimeError",
              True, str(e))
    try:
        bad_lens = list(lay.dst_lens)
        bad_lens[0] += 512
        mgr.send_kvcache("s1", SRC_PAGES, lay.dst_ptrs, DST_PAGES, EXECUTOR,
                         dst_attn_tp_size=dst_tp, dst_tp_rank=0,
                         dst_kv_item_lens=bad_lens)
        check("Bneg-guard", "target stride mismatch raises RuntimeError",
              False, "no exception raised")
    except RuntimeError as e:
        check("Bneg-guard", "target stride mismatch raises RuntimeError",
              True, str(e))


def scenario_c(conn, torch_fallback=False):
    src_tp, dst_tp = 2, 4
    tag = "C-torch" if torch_fallback else "C"
    for engine_rank, dst_tp_rank in [(0, 0), (1, 3)]:
        run_b_pair(conn, f"{tag} rank{engine_rank}->d{dst_tp_rank}",
                   src_tp, dst_tp, engine_rank, dst_tp_rank,
                   use_staging=True, staging_torch=torch_fallback)


def scenario_e(conn, label):
    src_tp, dst_tp = 4, 4
    lay = Layout(src_tp, dst_tp)
    mgr = make_manager(conn, "DFLASH", src_tp, engine_rank=0, lay=lay)
    rec = Recorder(vm_src=lay.vm_src)
    mgr._transfer_data = rec
    ret = mgr.send_kvcache(
        "s1", SRC_PAGES, lay.dst_ptrs, DST_PAGES, EXECUTOR,
        dst_layer_ids=DST_LAYER_IDS, dst_kv_item_len=TARGET_ITEM_LEN,
        dst_attn_tp_size=dst_tp,
    )
    check(label, "no raise on equal TP", ret == 0, f"ret={ret}")
    oob = rec.replay_into(lay.vm_dst)
    check(label, "no out-of-region writes", oob == 0, f"oob={oob}")
    verify_target_content(label, lay)
    n_bad = 0
    for j in range(NUM_DRAFT):
        gj = TARGET_ENTRIES + j
        for pos, d in enumerate(DST_LIST):
            got = lay.vm_dst.read(
                lay.dst_ptrs[gj] + d * lay.dst_draft_item, lay.dst_draft_item)
            if got != lay.expected_draft_page(j, SRC_LIST[pos], 0, 0):
                n_bad += 1
    check(label, "draft byte-exact equal-TP", n_bad == 0, f"bad={n_bad}")


def scenario_f(conn):
    src_tp, dst_tp = 2, 8
    for engine_rank, dst_tp_rank in [(0, 0), (0, 3), (1, 4), (1, 7)]:
        run_b_pair(conn, f"F rank{engine_rank}->d{dst_tp_rank}",
                   src_tp, dst_tp, engine_rank, dst_tp_rank)
    for engine_rank, dst_tp_rank in [(0, 4), (1, 0)]:
        run_b_pair(conn, f"Fneg-owner rank{engine_rank}->d{dst_tp_rank}",
                   src_tp, dst_tp, engine_rank, dst_tp_rank)
    run_b_pair(conn, "F-staging rank0->d3", src_tp, dst_tp, 0, 3,
               use_staging=True)


SCENARIOS = {
    "A": lambda conn: scenario_a_or_d(conn, "A", "DFLASH"),
    "D": lambda conn: scenario_a_or_d(conn, "D", "dflash"),
    "B": scenario_b,
    "C": lambda conn: scenario_c(conn, torch_fallback=False),
    "C-torch": lambda conn: scenario_c(conn, torch_fallback=True),
    "E": lambda conn: scenario_e(conn, "E"),
    "E-base": lambda conn: scenario_e(conn, "E-base"),
    "F": scenario_f,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="path to sglang source tree")
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    log(f"tree={args.tree} scenario={args.scenario}")
    log(f"python={sys.executable}")
    import torch

    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"devices={torch.cuda.device_count()}")

    conn = load_conn(args.tree)
    try:
        SCENARIOS[args.scenario](conn)
    except Exception:
        check(args.scenario, "scenario completed without exception", False,
              traceback.format_exc())

    n_pass = sum(1 for c in CHECKS if c[2])
    n_fail = len(CHECKS) - n_pass
    log(f"SUMMARY scenario={args.scenario} pass={n_pass} fail={n_fail}")
    for scen, name, ok, detail in CHECKS:
        log(f"  {'PASS' if ok else 'FAIL'} {scen}: {name}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(
                dict(scenario=args.scenario, tree=args.tree,
                     python=sys.executable, torch=torch.__version__,
                     conn_file=conn.__file__, pass_count=n_pass,
                     fail_count=n_fail,
                     checks=[dict(scenario=s, name=n, ok=o, detail=d)
                             for s, n, o, d in CHECKS]),
                f, indent=2)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
