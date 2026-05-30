# Grand Pattern Fibonacci Dual-Direction Architecture
# Comprehensive test suite

from collections import DynamicVector
from src.types import *
from src.ops import *


var pass_count = 0
var fail_count = 0
var total = 0


fn assert_fn(cond: Bool, test_name: String):
    total += 1
    if cond:
        pass_count += 1
        print("  PASS: ", test_name)
    else:
        fail_count += 1
        print("  FAIL: ", test_name)


fn test_tick_updates_perception_db():
    var r = Room(1)
    var reading = Embedding()
    reading.data = SIMD[DType, EMBED_DIM](0.0)
    reading.data[0] = 1.0

    let (err, surprise) = tick_room(r, reading, 1.0, 42, 0.5)

    assert_fn(r.perception_db.count() == 1, "tick updates perception DB count")
    assert_fn(r.perception_db.entries[0].emb.data[0] == 1.0, "tick stores correct embedding")
    assert_fn(r.perception_db.entries[0].sensor_id == 42, "tick stores correct sensor_id")


fn test_predict_generates_embedding():
    var r = Room(1)
    r.vibe.position = Embedding(SIMD[DType, EMBED_DIM](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    r.vibe.velocity = Embedding(SIMD[DType, EMBED_DIM](0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    r.vibe.acceleration = Embedding(SIMD[DType, EMBED_DIM](0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))

    let pred = predict(r)

    assert_fn(pred.data[0] > 0.0, "predict generates non-zero embedding")
    # position + velocity + 0.5*acceleration = 1.0 + 0.1 + 0.005 = 1.105
    assert_fn(abs(pred.data[0] - 1.105) < 1.0e-10, "predict computes correct value")


fn test_balance_check_passes():
    var r = Room(1)
    let t = Tick(1.0, 1, Embedding(), 1.0)

    r.perception_db.push(t)
    r.prediction_db.push(t)
    r.perception_db.push(t)
    r.prediction_db.push(t)

    assert_fn(balance_check(r), "balance check passes when equal")


fn test_balance_check_fails():
    var r = Room(1)
    let t = Tick(1.0, 1, Embedding(), 1.0)

    r.perception_db.push(t)
    r.perception_db.push(t)
    r.prediction_db.push(t)

    assert_fn(not balance_check(r), "balance check fails when unequal")


fn test_vibe_computation():
    var r = Room(1)
    for i in range(3):
        var emb = Embedding()
        emb.data[0] = DType(i + 1)
        r.perception_db.push(Tick(DType(i + 1), 1, emb, 1.0))

    compute_vibe(r)

    assert_fn(r.vibe.position.data[0] == 3.0, "vibe position is last entry")
    assert_fn(r.vibe.velocity.data[0] == 1.0, "vibe velocity is diff")
    assert_fn(r.vibe.acceleration.data[0] == 0.0, "vibe acceleration is diff-of-diff")
    assert_fn(r.vibe.strength == 3.0, "vibe strength is entry count")


fn test_merge_reduces_count():
    var db = TickDB()
    let emb = Embedding()
    let emb_data = SIMD[DType, EMBED_DIM](0.0)
    var e = Embedding(emb_data)
    e.data[0] = 1.0

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 1.0))
    db.push(Tick(3.0, 1, e, 1.0))

    let merged = merge_similar(db, 0.99)

    assert_fn(merged > 0, "merge found similar entries")
    assert_fn(db.count() < 3, "merge reduced count")


fn test_decay_reduces_strengths():
    var db = TickDB()
    let e = Embedding()

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 1.0))

    decay(db, 0.9)

    assert_fn(db.entries[0].strength < 1.0, "decay reduces strength")
    assert_fn(abs(db.entries[0].strength - 0.9) < 1.0e-10, "decay by correct amount")


fn test_prune_removes_weak():
    var db = TickDB()
    let e = Embedding()

    db.push(Tick(1.0, 1, e, 1.0))
    db.push(Tick(2.0, 1, e, 0.01))
    db.push(Tick(3.0, 1, e, 0.5))

    let pruned = prune(db, 0.1)

    assert_fn(pruned == 1, "prune removes exactly 1 weak entry")
    assert_fn(db.count() == 2, "prune leaves 2 entries")


fn test_full_gc_cycle():
    var r = Room(1)
    for i in range(5):
        var reading = Embedding()
        reading.data[0] = DType(i + 1)
        tick_room(r, reading, DType(i + 1), 1, 0.5)

    let report = gc(r, 0.99, 0.8, 0.5)

    assert_fn(balance_check(r), "GC maintains balance")
    assert_fn(report.merged >= 0, "GC returns merge count")
    assert_fn(report.pruned >= 0, "GC returns prune count")


fn test_cross_room_correlation():
    var a = Room(1)
    var b = Room(2)

    a.vibe.position = Embedding()
    a.vibe.position.data[0] = 1.0
    b.vibe.position = Embedding()
    b.vibe.position.data[0] = 1.0

    assert_fn(abs(correlate(a, b) - 1.0) < 1.0e-10, "identical vibes correlate at 1.0")

    b.vibe.position.data[0] = 0.0
    b.vibe.position.data[1] = 1.0

    assert_fn(abs(correlate(a, b)) < 1.0e-10, "orthogonal vibes correlate at 0.0")


fn test_murmur_between_rooms():
    var a = Room(1)
    var b = Room(2)

    a.vibe.position = Embedding()
    a.vibe.position.data[0] = 1.0
    b.vibe.position = Embedding()

    murmur(a, b, 0.5)

    assert_fn(abs(b.vibe.position.data[0] - 0.5) < 1.0e-10, "murmur blends vibe position")


fn test_graph_construction():
    var graph = CellularGraph()

    graph.add_room(1)
    graph.add_room(2)
    graph.add_room(3)

    assert_fn(len(graph.rooms) == 3, "graph has 3 rooms")

    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 0.5)

    assert_fn(len(graph.edges) == 2, "graph has 2 edges")
    assert_fn(graph.edges[0].from_id == 1, "edge 1 from correct")
    assert_fn(graph.edges[1].to_id == 3, "edge 2 to correct")


fn test_tick_propagation():
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

    assert_fn(
        graph.rooms[1].vibe.position.data[0] > 0.0,
        "tick propagation influences connected room via murmur",
    )


fn main():
    test_tick_updates_perception_db()
    test_predict_generates_embedding()
    test_balance_check_passes()
    test_balance_check_fails()
    test_vibe_computation()
    test_merge_reduces_count()
    test_decay_reduces_strengths()
    test_prune_removes_weak()
    test_full_gc_cycle()
    test_cross_room_correlation()
    test_murmur_between_rooms()
    test_graph_construction()
    test_tick_propagation()

    print("")
    print("Results: ", pass_count, " passed, ", fail_count, " failed")
    if fail_count > 0:
        print("FAIL: Some tests failed")
    else:
        print("ALL TESTS PASSED")
