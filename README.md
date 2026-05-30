# Grand Pattern Fibonacci Dual-Direction Architecture - Mojo

Mojo 🔥 implementation of the Grand Pattern cellular graph system, leveraging Mojo's SIMD and vectorization features.

## Architecture

The core system is a cellular graph where each cell (room) maintains:
- **Perception DB (Z_in)**: incoming sensor embeddings
- **Prediction DB (Z_out)**: predicted future embeddings
- **JEPA mapping**: cross-DB comparison computing prediction error (surprise)
- **Double-entry bookkeeping**: every tick updates BOTH databases, must balance
- **Vibe**: (position, velocity, acceleration) tuple on the embedding manifold
- **GC**: 3-phase (merge similar → decay old → prune weak)
- **Cellular graph**: rooms as nodes, algorithms as edges, murmur as gossip protocol

## Building

```bash
mojo build tests/test_gp.mojo -o test_gp
```

## Testing

```bash
mojo run tests/test_gp.mojo
```

## Project Structure

- `src/types.mojo` - Core type definitions (Embedding, Tick, Vibe, Room, CellularGraph)
- `src/ops.mojo` - Core operations (tick, predict, balance_check, compute_vibe, gc, murmur, correlate)
- `tests/test_gp.mojo` - Comprehensive test suite (13 tests)

## Requirements

- Mojo SDK (Modular language)
- No external dependencies

## Mojo Features Used

- `SIMD[DType, EMBED_DIM]` for native vectorized embedding operations
- `struct` (value types) for Embedding, Tick, Vibe
- `DynamicVector` for dynamic collections
- Direct SIMD arithmetic for embedding operations
