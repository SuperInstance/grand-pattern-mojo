# Grand Pattern Fibonacci Dual-Direction Architecture
# Core operations
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`; `inout` param -> `mut`
#   - `from` is a keyword -> parameter renamed `from_room` (murmur)
#   - `math.sqrt` via `from std import math`
# Semantics unchanged from originals at git edcb28b.

from std import sys
from std import math
from src.types import *


def cosine_similarity(a: Embedding, b: Embedding) -> Float64:
    var dot_p = (a.data * b.data).reduce_add()
    var na = a.norm()
    var nb = b.norm()
    if na < 1.0e-12 or nb < 1.0e-12:
        return 0.0
    return dot_p / (na * nb)


def cosine_distance(a: Embedding, b: Embedding) -> Float64:
    return 1.0 - cosine_similarity(a, b)


def predict(room: Room) -> Embedding:
    # p + v + 0.5*a
    return Embedding(
        room.vibe.position.data + room.vibe.velocity.data + room.vibe.acceleration.data * 0.5
    )


def compute_vibe(mut room: Room):
    var n = room.perception_db.count()

    if n == 0:
        # field-wise reset (whole-struct `room.vibe = Vibe()` through a mut
        # ref is rejected on this nightly: "destroyed out of the middle")
        room.vibe.position = Embedding()
        room.vibe.velocity = Embedding()
        room.vibe.acceleration = Embedding()
        room.vibe.strength = 0.0
        return

    # Position = last entry
    room.vibe.position = room.perception_db.entries[n - 1].emb

    if n >= 2:
        room.vibe.velocity = Embedding(
            room.perception_db.entries[n - 1].emb.data -
            room.perception_db.entries[n - 2].emb.data
        )
    else:
        room.vibe.velocity = Embedding()

    if n >= 3:
        var prev_diff = SIMD[DType.float64, 8](
            room.perception_db.entries[n - 2].emb.data -
            room.perception_db.entries[n - 3].emb.data
        )
        room.vibe.acceleration = Embedding(
            room.perception_db.entries[n - 1].emb.data -
            room.perception_db.entries[n - 2].emb.data - prev_diff
        )
    else:
        room.vibe.acceleration = Embedding()

    room.vibe.strength = Float64(n)


def tick_room(
    mut room: Room,
    reading: Embedding,
    timestamp: Float64,
    sensor_id: Int,
    threshold: Float64,
) -> Tuple[Float64, Bool]:
    # 1. Store perception in Z_in
    room.perception_db.push(Tick(timestamp, sensor_id, reading, 1.0))

    # 2. Generate prediction
    var predicted = predict(room)
    room.prediction_db.push(Tick(timestamp, sensor_id, predicted, 1.0))

    # 3. Compute prediction error
    var err = cosine_distance(reading, predicted)

    # 4. Surprise check
    var is_surprise = err > threshold

    # 5. Update vibe
    compute_vibe(room)

    return Tuple(err, is_surprise)


def balance_check(room: Room) -> Bool:
    return room.perception_db.count() == room.prediction_db.count()


def merge_similar(mut db: TickDB, threshold: Float64) -> Int:
    var n = db.count()
    var merged = 0
    var alive = List[Bool]()
    for _ in range(n):
        alive.append(True)

    for i in range(n):
        if not alive[i]:
            continue
        for j in range(i + 1, n):
            if not alive[j]:
                continue
            if cosine_similarity(db.entries[i].emb, db.entries[j].emb) > threshold:
                db.entries[i] = Tick(
                    db.entries[i].timestamp,
                    db.entries[i].sensor_id,
                    Embedding((db.entries[i].emb.data + db.entries[j].emb.data) * 0.5),
                    db.entries[i].strength + db.entries[j].strength,
                )
                alive[j] = False
                merged += 1

    # Compact
    var new_entries = List[Tick]()
    for i in range(n):
        if alive[i]:
            new_entries.append(db.entries[i])

    db._count = len(new_entries)
    db.entries = new_entries^
    return merged


def decay(mut db: TickDB, rate: Float64):
    for i in range(db.count()):
        var e = db.entries[i]
        db.entries[i] = Tick(e.timestamp, e.sensor_id, e.emb, e.strength * rate)


def prune(mut db: TickDB, min_strength: Float64) -> Int:
    var pruned = 0
    var new_entries = List[Tick]()
    for i in range(db.count()):
        if db.entries[i].strength >= min_strength:
            new_entries.append(db.entries[i])
        else:
            pruned += 1

    db._count = len(new_entries)
    db.entries = new_entries^
    return pruned


def gc(
    mut room: Room,
    merge_threshold: Float64,
    decay_rate: Float64,
    min_strength: Float64,
) -> GCReport:
    var report = GCReport()

    # Phase 1: Merge similar
    report.merged = merge_similar(room.perception_db, merge_threshold)
    report.merged += merge_similar(room.prediction_db, merge_threshold)

    # Phase 2: Decay
    decay(room.perception_db, decay_rate)
    decay(room.prediction_db, decay_rate)
    report.decayed = room.perception_db.count() + room.prediction_db.count()

    # Phase 3: Prune weak
    report.pruned = prune(room.perception_db, min_strength)
    report.pruned += prune(room.prediction_db, min_strength)

    # Rebalance
    var target = min(room.perception_db.count(), room.prediction_db.count())
    while room.perception_db.count() > target:
        var _ = room.perception_db.entries.pop()
        room.perception_db._count -= 1
    while room.prediction_db.count() > target:
        var _ = room.prediction_db.entries.pop()
        room.prediction_db._count -= 1

    return report


def murmur_pos(mut to_pos: Embedding, from_pos: Embedding, influence: Float64):
    # Blend helper shared by murmur() and propagate_tick() — same formula as
    # the original murmur, factored out so propagate_tick can mutate a room
    # inside the graph without aliasing two subscripts of the same List.
    to_pos = Embedding(
        to_pos.data * (1.0 - influence) + from_pos.data * influence
    )


def murmur(from_room: Room, mut to: Room, influence: Float64):
    murmur_pos(to.vibe.position, from_room.vibe.position, influence)


def correlate(room_a: Room, room_b: Room) -> Float64:
    return cosine_similarity(room_a.vibe.position, room_b.vibe.position)


def propagate_tick(
    mut graph: CellularGraph,
    from_room_idx: Int,
    reading: Embedding,
    timestamp: Float64,
    sensor_id: Int,
    threshold: Float64,
    murmur_influence: Float64,
):
    # NOTE: no Room copies (Room is not implicitly copyable on this nightly);
    # mutation flows through List subscript references — same net semantics
    # as the original (from_room was read-only there).
    var from_pos = graph.rooms[from_room_idx].vibe.position
    for i in range(len(graph.edges)):
        if graph.edges[i].from_id == graph.rooms[from_room_idx].id:
            var to_idx = graph.find_room(graph.edges[i].to_id)
            if to_idx >= 0:
                murmur_pos(graph.rooms[to_idx].vibe.position, from_pos, murmur_influence)
