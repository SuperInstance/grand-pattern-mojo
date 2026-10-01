# Grand Pattern Fibonacci Dual-Direction Architecture
# Comprehensive test suite
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - module-level `var` counters are BANNED ("global variables are not
#     supported") -> counters moved into a `Counters` struct passed as `mut`
#   - `SIMD[DType, EMBED_DIM]` -> `SIMD[DType.float64, 8]` (file-scope alias gone)
# Test bodies otherwise unchanged from originals at git edcb28b.

from std import sys
from src.types import *
from src.ops import *


struct Counters(ImplicitlyCopyable):
    var total: Int
    var pass_count: Int
    var fail_count: Int

    def __init__(out self):
        self.total = 0
        self.pass_count = 0
        self.fail_count = 0

    def record(mut self, cond: Bool, test_name: String):
        self.total += 1
        if cond:
            self.pass_count += 1
            print("  PASS: ", test_name)
        else:
            self.fail_count += 1
            print("  FAIL: ", test_name)


def test_tick_updates_perception_db(mut c: Counters):
    var r = Room(1)
    var reading = Embedding()
    reading.data = SIMD[DType.float64, 8](0.0)
    reading.data[0] = 1.0

    _ = tick_room(r, reading, 1.0, 42, 0.5)

    c.record(r.perception_db.count() == 1, "tick updates perception DB count")
    c.record(r.perception_db.entries[0].emb.data[0] == 1.0, "tick stores correct embedding")
    c.record(r.perception_db.entries[0].sensor_id == 42, "tick stores correct sensor_id")


def test_predict_generates_embedding(mut c: Counters):
    var r = Room(1)
    r.vibe.position = Embedding(SIMD[DType.float64, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    r.vibe.velocity = Embedding(SIMD[DType.float64, 8](0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    r.vibe.acceleration = Embedding(SIMD[DType.float64, 8](0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))

    var pred = predict(r)

    c.record(pred.data[0] > 0.0, "predict generates non-zero embedding")
    # position + velocity + 0.5*acceleration = 1.0 + 0.1 + 0.005 = 1.105
    c.record(abs(pred.data[0] - 1.105) < 1.0e-10, "predict computes correct value")


def test_balance_check_passes(mut c: Counters):
    var r = Room(1)
    var t = Tick(1.0, 1, Embedding(), 1.0)

    r.perception_db.push(t)
    r.prediction_db.push(t)
    r.perception_db.push(t)
    r.prediction_db.push(t)

    c.record(balance_check(r), "balance check passes when equal")


def test_balance_check_fails(mut c: Counters):
    var r = Room(1)
    var t = Tick(1.0, 1, Embedding(), 1.0)

    r.perception_db.push(t)
    r.perception_db.push(t)
    r.prediction_db.push(t)

    c.record(not balance_check(r), "balance check fails when unequal")


def test_vibe_computation(mut c: Counters):
    var r = Room(1)
    for i in range(3):
        var emb = Embedding()
        emb.data[0] = Float64(i + 1)
        r.perception_db.push(Tick(Float64(i + 1), 1, emb, 1.0))

    compute_vibe(r)

    c.record(r.vibe.position.data[0] == 3.0, "vibe position is last entry")
    c.record(r.vibe.velocity.data[0] == 1.0, "vibe velocity is diff")
    c.record(r.vibe.acceleration.data[0] == 0.0, "vibe acceleration is diff-of-diff")
    c.record(r.vibe.strength == 3.0, "vibe strength is entry count")


def test_merge_reduces_count(mut c: Counters):
    var db = TickDB()
    var e = Embedding()
    e.data[0] = 1.0

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 1.0))
    db.push(Tick(3.0, 1, e, 1.0))

    var merged = merge_similar(db, 0.99)

    c.record(merged > 0, "merge found similar entries")
    c.record(db.count() < 3, "merge reduced count")


def test_decay_reduces_strengths(mut c: Counters):
    var db = TickDB()
    var e = Embedding()

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 1.0))

    decay(db, 0.9)

    c.record(db.entries[0].strength < 1.0, "decay reduces strength")
    c.record(abs(db.entries[0].strength - 0.9) < 1.0e-10, "decay by correct amount")


def test_prune_removes_weak(mut c: Counters):
    var db = TickDB()
    var e = Embedding()

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 0.01))
    db.push(Tick(3.0, 1, e, 0.5))

    var pruned = prune(db, 0.1)

    c.record(pruned == 1, "prune removes exactly 1 weak entry")
    c.record(db.count() == 2, "prune leaves 2 entries")


def test_full_gc_cycle(mut c: Counters):
    var r = Room(1)
    for i in range(5):
        var reading = Embedding()
        reading.data[0] = Float64(i + 1)
        _ = tick_room(r, reading, Float64(i + 1), 1, 0.5)

    var report = gc(r, 0.99, 0.8, 0.5)

    c.record(balance_check(r), "GC maintains balance")
    c.record(report.merged >= 0, "GC returns merge count")
    c.record(report.pruned >= 0, "GC returns prune count")


def test_cross_room_correlation(mut c: Counters):
    var a = Room(1)
    var b = Room(2)

    a.vibe.position = Embedding()
    a.vibe.position.data[0] = 1.0
    b.vibe.position = Embedding()
    b.vibe.position.data[0] = 1.0

    c.record(abs(correlate(a, b) - 1.0) < 1.0e-10, "identical vibes correlate at 1.0")

    b.vibe.position.data[0] = 0.0
    b.vibe.position.data[1] = 1.0

    c.record(abs(correlate(a, b)) < 1.0e-10, "orthogonal vibes correlate at 0.0")


def test_murmur_between_rooms(mut c: Counters):
    var a = Room(1)
    var b = Room(2)

    a.vibe.position = Embedding()
    a.vibe.position.data[0] = 1.0
    b.vibe.position = Embedding()

    murmur(a, b, 0.5)

    c.record(abs(b.vibe.position.data[0] - 0.5) < 1.0e-10, "murmur blends vibe position")


def test_graph_construction(mut c: Counters):
    var graph = CellularGraph()

    graph.add_room(1)
    graph.add_room(2)
    graph.add_room(3)

    c.record(len(graph.rooms) == 3, "graph has 3 rooms")

    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 0.5)

    c.record(len(graph.edges) == 2, "graph has 2 edges")
    c.record(graph.edges[0].from_id == 1, "edge 1 from correct")
    c.record(graph.edges[1].to_id == 3, "edge 2 to correct")


def test_tick_propagation(mut c: Counters):
    var graph = CellularGraph()
    graph.add_room(1)
    graph.add_room(2)
    graph.add_edge(1, 2, 1.0)

    graph.rooms[0].vibe.position = Embedding()
    graph.rooms[0].vibe.position.data[0] = 1.0
    graph.rooms[1].vibe.position = Embedding()

    var reading = Embedding()
    reading.data[0] = 2.0

    propagate_tick(graph, 0, reading, 1.0, 1, 0.5, 0.5)

    c.record(
        graph.rooms[1].vibe.position.data[0] > 0.0,
        "tick propagation influences connected room via murmur",
    )


def main():
    var c = Counters()
    test_tick_updates_perception_db(c)
    test_predict_generates_embedding(c)
    test_balance_check_passes(c)
    test_balance_check_fails(c)
    test_vibe_computation(c)
    test_merge_reduces_count(c)
    test_decay_reduces_strengths(c)
    test_prune_removes_weak(c)
    test_full_gc_cycle(c)
    test_cross_room_correlation(c)
    test_murmur_between_rooms(c)
    test_graph_construction(c)
    test_tick_propagation(c)

    print("")
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count > 0:
        print("FAIL: Some tests failed")
    else:
        print("ALL TESTS PASSED")
