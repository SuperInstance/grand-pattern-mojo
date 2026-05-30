# Grand Pattern Fibonacci Dual-Direction Architecture
# Core operations

from math import sqrt
from collections import DynamicVector
from src.types import *


fn cosine_similarity(a: Embedding, b: Embedding) -> DType:
    let dot_p = (a.data * b.data).reduce_add()
    let na = a.norm()
    let nb = b.norm()
    if na < 1.0e-12 or nb < 1.0e-12:
        return 0.0
    return dot_p / (na * nb)


fn cosine_distance(a: Embedding, b: Embedding) -> DType:
    return 1.0 - cosine_similarity(a, b)


fn predict(room: Room) -> Embedding:
    # p + v + 0.5*a
    return Embedding(
        room.vibe.position.data + room.vibe.velocity.data + room.vibe.acceleration.data * 0.5
    )


fn compute_vibe(inout room: Room):
    let n = room.perception_db.count()

    if n == 0:
        room.vibe = Vibe()
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
        let prev_diff = SIMD[DType, EMBED_DIM](
            room.perception_db.entries[n - 2].emb.data -
            room.perception_db.entries[n - 3].emb.data
        )
        room.vibe.acceleration = Embedding(
            room.perception_db.entries[n - 1].emb.data -
            room.perception_db.entries[n - 2].emb.data - prev_diff
        )
    else:
        room.vibe.acceleration = Embedding()

    room.vibe.strength = DType(n)


fn tick_room(
    inout room: Room,
    reading: Embedding,
    timestamp: DType,
    sensor_id: Int,
    threshold: DType,
) -> Tuple[DType, Bool]:
    # 1. Store perception in Z_in
    room.perception_db.push(Tick(timestamp, sensor_id, reading, 1.0))

    # 2. Generate prediction
    let predicted = predict(room)
    room.prediction_db.push(Tick(timestamp, sensor_id, predicted, 1.0))

    # 3. Compute prediction error
    let err = cosine_distance(reading, predicted)

    # 4. Surprise check
    let is_surprise = err > threshold

    # 5. Update vibe
    compute_vibe(room)

    return Tuple(err, is_surprise)


fn balance_check(room: Room) -> Bool:
    return room.perception_db.count() == room.prediction_db.count()


fn merge_similar(inout db: TickDB, threshold: DType) -> Int:
    let n = db.count()
    var merged = 0
    var alive = DynamicVector[Bool]()
    for i in range(n):
        alive.push_back(True)

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
    var new_entries = DynamicVector[Tick]()
    for i in range(n):
        if alive[i]:
            new_entries.push_back(db.entries[i])

    db.entries = new_entries
    db._count = len(new_entries)
    return merged


fn decay(inout db: TickDB, rate: DType):
    for i in range(db.count()):
        let e = db.entries[i]
        db.entries[i] = Tick(e.timestamp, e.sensor_id, e.emb, e.strength * rate)


fn prune(inout db: TickDB, min_strength: DType) -> Int:
    var pruned = 0
    var new_entries = DynamicVector[Tick]()
    for i in range(db.count()):
        if db.entries[i].strength >= min_strength:
            new_entries.push_back(db.entries[i])
        else:
            pruned += 1

    db.entries = new_entries
    db._count = len(new_entries)
    return pruned


fn gc(
    inout room: Room,
    merge_threshold: DType,
    decay_rate: DType,
    min_strength: DType,
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
    let target = min(room.perception_db.count(), room.prediction_db.count())
    while room.perception_db.count() > target:
        room.perception_db.entries.pop_back()
        room.perception_db._count -= 1
    while room.prediction_db.count() > target:
        room.prediction_db.entries.pop_back()
        room.prediction_db._count -= 1

    return report


fn murmur(from: Room, inout to: Room, influence: DType):
    to.vibe.position = Embedding(
        to.vibe.position.data * (1.0 - influence) + from.vibe.position.data * influence
    )


fn correlate(room_a: Room, room_b: Room) -> DType:
    return cosine_similarity(room_a.vibe.position, room_b.vibe.position)


fn propagate_tick(
    inout graph: CellularGraph,
    from_room_idx: Int,
    reading: Embedding,
    timestamp: DType,
    sensor_id: Int,
    threshold: DType,
    murmur_influence: DType,
):
    let from_room = graph.rooms[from_room_idx]
    for i in range(len(graph.edges)):
        if graph.edges[i].from_id == from_room.id:
            let to_idx = graph.find_room(graph.edges[i].to_id)
            if to_idx >= 0:
                murmur(from_room, graph.rooms[to_idx], murmur_influence)
