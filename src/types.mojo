# Grand Pattern Fibonacci Dual-Direction Architecture
# Core type definitions

from math import sqrt

alias EMBED_DIM = 8
alias DType = Float64


@value
struct Embedding:
    var data: SIMD[DType, EMBED_DIM]

    fn __init__(inout self):
        self.data = SIMD[DType, EMBED_DIM](0.0)

    fn __init__(inout self, values: SIMD[DType, EMBED_DIM]):
        self.data = values

    fn norm(self) -> DType:
        return sqrt((self.data * self.data).reduce_add())

    fn __add__(self, other: Embedding) -> Embedding:
        return Embedding(self.data + other.data)

    fn __sub__(self, other: Embedding) -> Embedding:
        return Embedding(self.data - other.data)

    fn __mul__(self, scalar: DType) -> Embedding:
        return Embedding(self.data * scalar)

    fn zero() -> Embedding:
        return Embedding()


@value
struct Tick:
    var timestamp: DType
    var sensor_id: Int
    var emb: Embedding
    var strength: DType

    fn __init__(inout self):
        self.timestamp = 0.0
        self.sensor_id = 0
        self.emb = Embedding()
        self.strength = 1.0

    fn __init__(inout self, ts: DType, sid: Int, emb: Embedding, str: DType = 1.0):
        self.timestamp = ts
        self.sensor_id = sid
        self.emb = emb
        self.strength = str


@value
struct Vibe:
    var position: Embedding
    var velocity: Embedding
    var acceleration: Embedding
    var strength: DType

    fn __init__(inout self):
        self.position = Embedding()
        self.velocity = Embedding()
        self.acceleration = Embedding()
        self.strength = 1.0


@value
struct GCReport:
    var merged: Int
    var decayed: Int
    var pruned: Int

    fn __init__(inout self):
        self.merged = 0
        self.decayed = 0
        self.pruned = 0


struct TickDB:
    var entries: DynamicVector[Tick]
    var _count: Int

    fn __init__(inout self):
        self.entries = DynamicVector[Tick]()
        self._count = 0

    fn push(inout self, entry: Tick):
        self.entries.push_back(entry)
        self._count += 1

    fn count(self) -> Int:
        return self._count

    fn last(self) -> Tick:
        if self._count > 0:
            return self.entries[self._count - 1]
        return Tick()


struct Room:
    var id: Int
    var perception_db: TickDB
    var prediction_db: TickDB
    var vibe: Vibe

    fn __init__(inout self, id: Int = 0):
        self.id = id
        self.perception_db = TickDB()
        self.prediction_db = TickDB()
        self.vibe = Vibe()


@value
struct Edge:
    var from_id: Int
    var to_id: Int
    var weight: DType

    fn __init__(inout self, from_id: Int, to_id: Int, weight: DType = 1.0):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight


struct CellularGraph:
    var rooms: DynamicVector[Room]
    var edges: DynamicVector[Edge]

    fn __init__(inout self):
        self.rooms = DynamicVector[Room]()
        self.edges = DynamicVector[Edge]()

    fn add_room(inout self, id: Int):
        self.rooms.push_back(Room(id))

    fn add_edge(inout self, from_id: Int, to_id: Int, weight: DType = 1.0):
        self.edges.push_back(Edge(from_id, to_id, weight))

    fn find_room(self, id: Int) -> Int:
        for i in range(len(self.rooms)):
            if self.rooms[i].id == id:
                return i
        return -1
