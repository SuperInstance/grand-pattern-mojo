# USERMANUAL — grand-pattern-mojo

*Verified on-box 2026-10-01, Mojo 1.2.0.dev2026100105, Linux/WSL2 x64, CPU-only.*

## What this is

A Mojo 🔥 implementation of the **Grand Pattern cellular graph**: rooms
(cells) each carry two tick databases —

- **Perception DB (Z_in)** — incoming sensor embeddings
- **Prediction DB (Z_out)** — predicted future embeddings

— updated in strict double-entry balance (every tick pushes to both), with:

- **JEPA-style prediction error**: `predict = p + v + 0.5·a` (stale vibe,
  i.e. computed *after* storing the reading but *before* the vibe update);
  error = cosine distance(reading, predicted); error > threshold = surprise
- **Vibe**: (position, velocity, acceleration) kinematics on the embedding
  manifold, computed from the last 1/2/3 perception entries
- **3-phase GC**: merge similar (greedy pairwise, cosine > threshold) →
  decay (strength × rate) → prune (strength < min), then rebalance by
  popping the *tail* of the larger DB
- **Murmur**: gossip that blends a neighbor's vibe position toward this
  room's position, `to·(1−inf) + from·inf`, propagated along graph edges

Canonical spec = `src/ops.mojo` + `src/types.mojo` (the README prose is an
accurate summary). Scaffolding, not spec: `AGENT.md` (fleet identity),
`memory/JOURNAL.md` (placeholder), `.github/workflows/ci.yml` (echo stub).

## Quickstart

```bash
# toolchain on this box (mojo not on bare PATH):
export MODULAR_HOME=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/share/max
export PATH=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/bin:$PATH

cd grand-pattern-mojo

make test      # full suite, -D ASSERT=all, flags BEFORE filename
make build     # mojo build -I . tests/test_gp.mojo -o test_gp
make demo      # end-to-end receipt (examples/demo.mojo)
make parity    # demo vs Python oracle, bit-for-bit diff
```

`-I .` is **required** for every mojo invocation — see troubleshooting.

## Test suite

`tests/test_gp.mojo` — 13 test functions, **28 assertions**, self-reporting
(no global counters — see traps). Expected output ends:

```
Results:  28  passed,  0  failed
ALL TESTS PASSED
```

Verified this box: `make test` → 28/28 PASS, with and without `-D ASSERT=all`.
(The suite uses its own `Counters.record`, not `assert`; `ASSERT=all` changes
nothing here but is run anyway per fleet convention.)

## Parity contract

`python/oracle_gp.py` is a pure-Python float64 oracle mirroring ops.mojo op
for op. `examples/demo.mojo` runs a 12-case deterministic scenario (cosine
incl. zero-vector guard, predict, vibe on 3 readings, fresh-room tick,
merge, decay+prune, full 5-tick GC cycle incl. the pop-back rebalance that
keeps the zero vector, murmur, 3-room graph propagation, empty room) and
prints every Float64 as its IEEE754 bit pattern (`std.memory.bitcast`).

```bash
make parity
```

**Expected: `Parity: 30/30 lines bit-identical` / `PARITY OK — bit-for-bit`.**

Verified this box: 30/30 bit-identical, 2026-10-01 (outputs in `outputs/`).
Canonical side per spec: ops.mojo semantics (both sides agree anyway — same
IEEE754 double ops in the same order; `reduce_add` over 8 lanes matched
Python's sequential left-to-right accumulation exactly).

If you change any op formula: run `make parity` before committing. A
divergence means one side drifted — ops.mojo + oracle must be re-derived
together.

## This-box demo receipt (verbatim tail)

```
CASE gc5_merged 7
CASE gc5_decayed 3
CASE gc5_pruned 0
CASE gc5_balance True
CASE gc5_count 1
CASE gc5_s0 0x4010000000000000
CASE murmur  0x3fe0000000000000  0x0  0x0  0x0  0x0  0x0  0x0  0x0
CASE graph_pos0 0x3ff0000000000000
CASE graph_pos1 0x3fe0000000000000
CASE graph_pos2 0x0
CASE empty_strength 0x0
CASE empty_pos0 0x0
DEMO OK — grand-pattern-mojo end-to-end receipt
```

(Full receipt: `outputs/demo_mojo.txt` / `outputs/demo_python.txt`.)

## Mojo 1.2.0-dev traps — port notes & troubleshooting

Banked fleet traps re-confirmed here: flags after the source filename are
silently ignored (always flags-first); `len(String)` banned → `byte_length()`;
`from collections import` does not exist; prelude needs ≥1 `from std import`.
**New traps hit during this port:**

1. **Module-level `var` is banned.** `var x = 0` at file scope →
   `error: global variables are not supported; move this into a function
   body or use 'comptime'`. Test counters now live in a `Counters` struct
   passed as `mut`.
2. **Imports resolve relative to the main file's directory, not cwd.**
   `mojo run tests/test_gp.mojo` cannot see `src/`; you MUST pass
   `mojo run -I . tests/test_gp.mojo`. Symptom: `unable to locate module 'src'`.
3. **A package directory needs `__init__.mojo`.** `src/__init__.mojo` (may be
   comment-only) is what makes `from src.types import *` legal.
4. **`math` moved under std**: `from std import math`, call `math.sqrt`.
   Free functions too: `from std.memory import bitcast` (the compiler
   suggests this one if you try bare `bitcast`).
5. **`DynamicVector` is gone; use builtin `List`**: `.append()` not
   `.push_back()`, `.pop()` not `.pop_back()`.
6. **`List` is NOT `ImplicitlyCopyable`** — so any struct with a `List`
   field (TickDB, Room, CellularGraph) cannot be either, and an explicit
   `def __copyinit__` fails with `'None' has no attributes` (dunder not
   recognized on `def` in this nightly). Fix: never copy List-holders;
   mutate through `mut` refs and List subscripts (verified: mutation through
   subscript chains propagates in place).
7. **Aliasing guard**: passing two subscripts of the same List as
   (read, mut) args to one call is a compile error
   (`aliasing values passed immutably ... and passed mutably`). Fix: copy the
   small `Embedding` out first and mutate via `murmur_pos` helper.
8. **`^` transfer then read = uninitialized**: `db.entries = new_entries^`
   consumes `new_entries` — compute `len()` (and anything else you need)
   BEFORE the transfer.
9. **Whole-struct field overwrite through a `mut` ref is rejected**:
   `room.vibe = Vibe()` → `field 'room.vibe.position' destroyed out of the
   middle of a value`. Assign the fields individually.
10. **`Float64` has no `.bitcast` method** — use the free function:
    `bitcast[DType.uint64, 1](SIMD[DType.float64, 1](x))`.
11. **`hex(UInt64)`** prints `0x`-prefixed lowercase with NO sign-extension
    (`0xffffffffffffffff` is correct for all-bits-set).
12. **`print(True)` → `True`** (capitalized, like Python — not `true`).
13. **`pixi run mojo` only works in a dir with a pixi.toml** (e.g.
    quilt-mojo-lab). From this repo use the two exports in Quickstart.

## Repo layout

```
src/types.mojo        Embedding, Tick, Vibe, GCReport, TickDB, Room, Edge, CellularGraph
src/ops.mojo          cosine_*, predict, compute_vibe, tick_room, balance_check,
                      merge_similar, decay, prune, gc, murmur(_pos), correlate,
                      propagate_tick
tests/test_gp.mojo    13 tests / 28 assertions
examples/demo.mojo    end-to-end receipt + parity driver
python/oracle_gp.py   Python float64 oracle (bit-for-bit reference)
python/parity_check.py
outputs/              demo receipts (mojo + python)
```

Originals of every ported file are preserved in git history at commit
`edcb28b` (pre-port state); this port is mechanical only — formulas,
thresholds, and semantics unchanged.
