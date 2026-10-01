# Grand Pattern — end-to-end demo & parity driver.
#
# Runs a deterministic scenario through every op (no RNG) and prints a
# receipt. Float64 results are printed as IEEE754 bit patterns (via
# `from std.memory import bitcast`) so the Python oracle
# (python/oracle_gp.py) can be diffed BIT-FOR-BIT:
#
#   mojo run -I . examples/demo.mojo > outputs/demo_mojo.txt
#   python3 python/oracle_gp.py            > outputs/demo_python.txt
#   diff outputs/demo_mojo.txt outputs/demo_python.txt
#
# hex(UInt64) prints 0x-prefixed; the oracle driver normalizes both sides.

from std import sys
from std.memory import bitcast
from src.types import *
from src.ops import *


def bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64, 1](SIMD[DType.float64, 1](x))[0]


def emb(values: SIMD[DType.float64, 8]) -> Embedding:
    return Embedding(values)


def print_vec(name: String, e: Embedding):
    print(name, end="")
    for i in range(8):
        print(" ", hex(bits(e.data[i])), end="")
    print()


def main():
    # --- cosine similarity cases -----------------------------------------
    var a = emb(SIMD[DType.float64, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    print("CASE cos_identical", hex(bits(cosine_similarity(a, a))))
    var b = emb(SIMD[DType.float64, 8](0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    print("CASE cos_orthogonal", hex(bits(cosine_similarity(a, b))))
    var z = Embedding()
    print("CASE cos_zero", hex(bits(cosine_similarity(z, a))))

    # --- predict: p + v + 0.5*a -------------------------------------------
    var r = Room(1)
    r.vibe.position = emb(SIMD[DType.float64, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    r.vibe.velocity = emb(SIMD[DType.float64, 8](0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    r.vibe.acceleration = emb(SIMD[DType.float64, 8](0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))
    print_vec("CASE predict", predict(r))

    # --- vibe on 3 sequential readings ------------------------------------
    var v3 = Room(2)
    for i in range(3):
        var e = Embedding()
        e.data[0] = Float64(i + 1)
        v3.perception_db.push(Tick(Float64(i + 1), 7, e, 1.0))
    compute_vibe(v3)
    print_vec("CASE vibe3_pos", v3.vibe.position)
    print_vec("CASE vibe3_vel", v3.vibe.velocity)
    print_vec("CASE vibe3_acc", v3.vibe.acceleration)
    print("CASE vibe3_strength", hex(bits(v3.vibe.strength)))

    # --- single tick on a fresh room (err must be 1.0 via zero-guard) ------
    var r1 = Room(3)
    var reading = Embedding()
    reading.data[0] = 1.0
    var result = tick_room(r1, reading, 1.0, 42, 0.5)
    print("CASE tick1_err", hex(bits(result[0])))
    print("CASE tick1_surprise", result[1])

    # --- merge_similar on 3 identical embeddings ---------------------------
    var db = TickDB()
    var e1 = Embedding()
    e1.data[0] = 1.0
    db.push(Tick(1.0, 1, e1, 1.0))
    db.push(Tick(2.0, 1, e1, 1.0))
    db.push(Tick(3.0, 1, e1, 1.0))
    var merged_n = merge_similar(db, 0.99)
    print("CASE merge3_merged", merged_n)
    print("CASE merge3_count", db.count())
    print("CASE merge3_strength", hex(bits(db.entries[0].strength)))

    # --- decay + prune ------------------------------------------------------
    var db2 = TickDB()
    db2.push(Tick(1.0, 1, e1, 1.0))
    var e_weak = Embedding()
    e_weak.data[1] = 1.0
    db2.push(Tick(2.0, 1, e_weak, 0.01))
    var e_mid = Embedding()
    e_mid.data[2] = 1.0
    db2.push(Tick(3.0, 1, e_mid, 0.5))
    decay(db2, 0.9)
    var pruned_n = prune(db2, 0.1)
    print("CASE decay_prune_pruned", pruned_n)
    print("CASE decay_prune_count", db2.count())
    for i in range(db2.count()):
        print("CASE decay_prune_s", i, hex(bits(db2.entries[i].strength)))

    # --- full gc cycle: 5 sequential ticks ----------------------------------
    var g = Room(4)
    for i in range(5):
        var rd = Embedding()
        rd.data[0] = Float64(i + 1)
        _ = tick_room(g, rd, Float64(i + 1), 9, 0.5)
    var report = gc(g, 0.99, 0.8, 0.5)
    print("CASE gc5_merged", report.merged)
    print("CASE gc5_decayed", report.decayed)
    print("CASE gc5_pruned", report.pruned)
    print("CASE gc5_balance", balance_check(g))
    print("CASE gc5_count", g.perception_db.count())
    print("CASE gc5_s0", hex(bits(g.perception_db.entries[0].strength)))

    # --- murmur blend --------------------------------------------------------
    var ma = Room(5)
    ma.vibe.position = emb(SIMD[DType.float64, 8](1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    var mb = Room(6)
    murmur(ma, mb, 0.5)
    print_vec("CASE murmur", mb.vibe.position)

    # --- graph propagation over edges 1->2, 2->3 ----------------------------
    var graph = CellularGraph()
    graph.add_room(1)
    graph.add_room(2)
    graph.add_room(3)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 0.5)
    graph.rooms[0].vibe.position = emb(SIMD[DType.float64, 8](1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    var greading = Embedding()
    greading.data[0] = 2.0
    propagate_tick(graph, 0, greading, 1.0, 1, 0.5, 0.5)
    print("CASE graph_pos0", hex(bits(graph.rooms[0].vibe.position.data[0])))
    print("CASE graph_pos1", hex(bits(graph.rooms[1].vibe.position.data[0])))
    print("CASE graph_pos2", hex(bits(graph.rooms[2].vibe.position.data[0])))

    # --- empty/degenerate room ----------------------------------------------
    var empty = Room(7)
    compute_vibe(empty)
    print("CASE empty_strength", hex(bits(empty.vibe.strength)))
    print("CASE empty_pos0", hex(bits(empty.vibe.position.data[0])))

    print("DEMO OK — grand-pattern-mojo end-to-end receipt")
