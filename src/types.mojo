# Grand Pattern Fibonacci Dual-Direction Architecture
# Core type definitions
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`; `inout self` ctor -> `out self`;
#     mutating methods -> `mut self`
#   - `@value` removed -> `struct X(ImplicitlyCopyable)`
#   - file-scope `alias` removed (parse error) -> inline `SIMD[DType.float64, 8]`
#   - `from math import sqrt` -> `from std import math` (call as `math.sqrt`)
#   - `DynamicVector` -> builtin `List` (module 'collections' no longer exists)
# Originals verbatim at git edcb28b.

from std import sys
from std import math


struct Embedding(ImplicitlyCopyable):
    var data: SIMD[DType.float64, 8]

    def __init__(out self):
        self.data = SIMD[DType.float64, 8](0.0)

    def __init__(out self, values: SIMD[DType.float64, 8]):
        self.data = values

    def norm(self) -> Float64:
        return math.sqrt((self.data * self.data).reduce_add())

    def __add__(self, other: Embedding) -> Embedding:
        return Embedding(self.data + other.data)

    def __sub__(self, other: Embedding) -> Embedding:
        return Embedding(self.data - other.data)

    def __mul__(self, scalar: Float64) -> Embedding:
        return Embedding(self.data * scalar)

    def zero() -> Embedding:
        return Embedding()


struct Tick(ImplicitlyCopyable):
    var timestamp: Float64
    var sensor_id: Int
    var emb: Embedding
    var strength: Float64

    def __init__(out self):
        self.timestamp = 0.0
        self.sensor_id = 0
        self.emb = Embedding()
        self.strength = 1.0

    def __init__(out self, ts: Float64, sid: Int, emb: Embedding, strength: Float64 = 1.0):
        self.timestamp = ts
        self.sensor_id = sid
        self.emb = emb
        self.strength = strength


struct Vibe(ImplicitlyCopyable):
    var position: Embedding
    var velocity: Embedding
    var acceleration: Embedding
    var strength: Float64

    def __init__(out self):
        self.position = Embedding()
        self.velocity = Embedding()
        self.acceleration = Embedding()
        self.strength = 1.0


struct GCReport(ImplicitlyCopyable):
    var merged: Int
    var decayed: Int
    var pruned: Int

    def __init__(out self):
        self.merged = 0
        self.decayed = 0
        self.pruned = 0


# NOTE: TickDB, Room, CellularGraph are NOT ImplicitlyCopyable — they hold
# List[...] fields, and List is not implicitly copyable (nor does explicit
# __copyinit__ synthesis work on this nightly). Nothing in src/ or tests/
# copies them by value; all mutation goes through `mut` references.
struct TickDB:
    var entries: List[Tick]
    var _count: Int

    def __init__(out self):
        self.entries = List[Tick]()
        self._count = 0

    def push(mut self, entry: Tick):
        self.entries.append(entry)
        self._count += 1

    def count(self) -> Int:
        return self._count

    def last(self) -> Tick:
        if self._count > 0:
            return self.entries[self._count - 1]
        return Tick()


struct Room:
    var id: Int
    var perception_db: TickDB
    var prediction_db: TickDB
    var vibe: Vibe

    def __init__(out self, id: Int = 0):
        self.id = id
        self.perception_db = TickDB()
        self.prediction_db = TickDB()
        self.vibe = Vibe()


struct Edge(ImplicitlyCopyable):
    var from_id: Int
    var to_id: Int
    var weight: Float64

    def __init__(out self, from_id: Int, to_id: Int, weight: Float64 = 1.0):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight


struct CellularGraph:
    var rooms: List[Room]
    var edges: List[Edge]

    def __init__(out self):
        self.rooms = List[Room]()
        self.edges = List[Edge]()

    def add_room(mut self, id: Int):
        self.rooms.append(Room(id))

    def add_edge(mut self, from_id: Int, to_id: Int, weight: Float64 = 1.0):
        self.edges.append(Edge(from_id, to_id, weight))

    def find_room(self, id: Int) -> Int:
        for i in range(len(self.rooms)):
            if self.rooms[i].id == id:
                return i
        return -1
