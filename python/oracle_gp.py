#!/usr/bin/env python3
"""Grand Pattern — Python oracle for bit-for-bit parity with examples/demo.mojo.

Mirrors src/ops.mojo semantics exactly (float64 IEEE throughout, same op
order). Output lines match the Mojo demo; the driver below normalizes hex
tokens from both sides before diffing.

    python3 python/oracle_gp.py > outputs/demo_python.txt
    mojo run -I . examples/demo.mojo > outputs/demo_mojo.txt
    python3 python/parity_check.py outputs/demo_mojo.txt outputs/demo_python.txt
"""

import struct
import sys

DIM = 8


def hexbits(x: float) -> str:
    return "0x" + format(struct.unpack("<Q", struct.pack("<d", x))[0], "x")


def vec(*xs) -> list:
    out = [0.0] * DIM
    for i, x in enumerate(xs):
        out[i] = float(x)
    return out


def dot(a: list, b: list) -> float:
    s = 0.0
    for i in range(DIM):
        s += a[i] * b[i]
    return s


def norm(a: list) -> float:
    return dot(a, a) ** 0.5  # sqrt


def cosine_similarity(a: list, b: list) -> float:
    d = dot(a, b)
    na = norm(a)
    nb = norm(b)
    if na < 1.0e-12 or nb < 1.0e-12:
        return 0.0
    return d / (na * nb)


def cosine_distance(a: list, b: list) -> float:
    return 1.0 - cosine_similarity(a, b)


def predict(vibe: dict) -> list:
    return [vibe["position"][i] + vibe["velocity"][i] + vibe["acceleration"][i] * 0.5 for i in range(DIM)]


def compute_vibe(room: dict) -> None:
    n = len(room["perception"])
    if n == 0:
        room["vibe"] = {
            "position": vec(),
            "velocity": vec(),
            "acceleration": vec(),
            "strength": 0.0,
        }
        return
    pos = list(room["perception"][-1]["emb"])
    vel = vec()
    if n >= 2:
        vel = [room["perception"][-1]["emb"][i] - room["perception"][-2]["emb"][i] for i in range(DIM)]
    acc = vec()
    if n >= 3:
        prev_diff = [room["perception"][-2]["emb"][i] - room["perception"][-3]["emb"][i] for i in range(DIM)]
        acc = [room["perception"][-1]["emb"][i] - room["perception"][-2]["emb"][i] - prev_diff[i] for i in range(DIM)]
    room["vibe"] = {"position": pos, "velocity": vel, "acceleration": acc, "strength": float(n)}


def tick_room(room: dict, reading: list, timestamp: float, sensor_id: int, threshold: float):
    room["perception"].append({"ts": timestamp, "sid": sensor_id, "emb": list(reading), "strength": 1.0})
    predicted = predict(room["vibe"])  # stale vibe, as in ops.mojo
    room["prediction"].append({"ts": timestamp, "sid": sensor_id, "emb": predicted, "strength": 1.0})
    err = cosine_distance(reading, predicted)
    surprise = err > threshold
    compute_vibe(room)
    return err, surprise


def merge_similar(entries: list, threshold: float) -> int:
    n = len(entries)
    merged = 0
    alive = [True] * n
    for i in range(n):
        if not alive[i]:
            continue
        for j in range(i + 1, n):
            if not alive[j]:
                continue
            if cosine_similarity(entries[i]["emb"], entries[j]["emb"]) > threshold:
                entries[i] = {
                    "ts": entries[i]["ts"],
                    "sid": entries[i]["sid"],
                    "emb": [(entries[i]["emb"][k] + entries[j]["emb"][k]) * 0.5 for k in range(DIM)],
                    "strength": entries[i]["strength"] + entries[j]["strength"],
                }
                alive[j] = False
                merged += 1
    return merged, [e for i, e in enumerate(entries) if alive[i]]


def decay(entries: list, rate: float) -> None:
    for e in entries:
        e["strength"] = e["strength"] * rate


def prune(entries: list, min_strength: float):
    kept = [e for e in entries if e["strength"] >= min_strength]
    return len(entries) - len(kept), kept


def murmur_pos(to_pos: list, from_pos: list, influence: float) -> list:
    return [to_pos[i] * (1.0 - influence) + from_pos[i] * influence for i in range(DIM)]


def print_vec(name: str, e: list) -> None:
    print(name, *(" ".join([hexbits(x) for x in e])).split(" "))


def main() -> None:
    # --- cosine similarity cases ---
    a = vec(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    print("CASE cos_identical", hexbits(cosine_similarity(a, a)))
    b = vec(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    print("CASE cos_orthogonal", hexbits(cosine_similarity(a, b)))
    z = vec()
    print("CASE cos_zero", hexbits(cosine_similarity(z, a)))

    # --- predict ---
    vibe = {
        "position": vec(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        "velocity": vec(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        "acceleration": vec(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    }
    print_vec("CASE predict", predict(vibe))

    # --- vibe on 3 sequential readings ---
    v3 = {"perception": [], "prediction": [], "vibe": {}}
    for i in range(3):
        e = vec()
        e[0] = float(i + 1)
        v3["perception"].append({"ts": float(i + 1), "sid": 7, "emb": e, "strength": 1.0})
    compute_vibe(v3)
    print_vec("CASE vibe3_pos", v3["vibe"]["position"])
    print_vec("CASE vibe3_vel", v3["vibe"]["velocity"])
    print_vec("CASE vibe3_acc", v3["vibe"]["acceleration"])
    print("CASE vibe3_strength", hexbits(v3["vibe"]["strength"]))

    # --- single tick on a fresh room ---
    r1 = {"perception": [], "prediction": [], "vibe": {"position": vec(), "velocity": vec(), "acceleration": vec(), "strength": 1.0}}
    reading = vec()
    reading[0] = 1.0
    err, surprise = tick_room(r1, reading, 1.0, 42, 0.5)
    print("CASE tick1_err", hexbits(err))
    print("CASE tick1_surprise", surprise)

    # --- merge_similar on 3 identical embeddings ---
    e1 = vec()
    e1[0] = 1.0
    db = [
        {"ts": 1.0, "sid": 1, "emb": list(e1), "strength": 1.0},
        {"ts": 2.0, "sid": 1, "emb": list(e1), "strength": 1.0},
        {"ts": 3.0, "sid": 1, "emb": list(e1), "strength": 1.0},
    ]
    merged_n, db = merge_similar(db, 0.99)
    print("CASE merge3_merged", merged_n)
    print("CASE merge3_count", len(db))
    print("CASE merge3_strength", hexbits(db[0]["strength"]))

    # --- decay + prune ---
    e_weak = vec()
    e_weak[1] = 1.0
    e_mid = vec()
    e_mid[2] = 1.0
    db2 = [
        {"ts": 1.0, "sid": 1, "emb": list(e1), "strength": 1.0},
        {"ts": 2.0, "sid": 1, "emb": e_weak, "strength": 0.01},
        {"ts": 3.0, "sid": 1, "emb": e_mid, "strength": 0.5},
    ]
    decay(db2, 0.9)
    pruned_n, db2 = prune(db2, 0.1)
    print("CASE decay_prune_pruned", pruned_n)
    print("CASE decay_prune_count", len(db2))
    for i, e in enumerate(db2):
        print("CASE decay_prune_s", i, hexbits(e["strength"]))

    # --- full gc cycle: 5 sequential ticks ---
    g = {"perception": [], "prediction": [], "vibe": {"position": vec(), "velocity": vec(), "acceleration": vec(), "strength": 1.0}}
    for i in range(5):
        rd = vec()
        rd[0] = float(i + 1)
        tick_room(g, rd, float(i + 1), 9, 0.5)
    merged_p, g["perception"] = merge_similar(g["perception"], 0.99)
    merged_q, g["prediction"] = merge_similar(g["prediction"], 0.99)
    total_merged = merged_p + merged_q
    decay(g["perception"], 0.8)
    decay(g["prediction"], 0.8)
    total_decayed = len(g["perception"]) + len(g["prediction"])
    pruned_p, g["perception"] = prune(g["perception"], 0.5)
    pruned_q, g["prediction"] = prune(g["prediction"], 0.5)
    total_pruned = pruned_p + pruned_q
    target = min(len(g["perception"]), len(g["prediction"]))
    del g["perception"][target:]
    del g["prediction"][target:]
    balance = len(g["perception"]) == len(g["prediction"])
    print("CASE gc5_merged", total_merged)
    print("CASE gc5_decayed", total_decayed)
    print("CASE gc5_pruned", total_pruned)
    print("CASE gc5_balance", balance)
    print("CASE gc5_count", len(g["perception"]))
    print("CASE gc5_s0", hexbits(g["perception"][0]["strength"]))

    # --- murmur blend ---
    ma = vec()
    ma[0] = 1.0
    mb = murmur_pos(vec(), ma, 0.5)
    print_vec("CASE murmur", mb)

    # --- graph propagation ---
    pos0 = ma
    pos1 = murmur_pos(vec(), pos0, 0.5)  # edge 1->2
    pos2 = vec()                          # edge 2->3 not traversed from room 1
    print("CASE graph_pos0", hexbits(pos0[0]))
    print("CASE graph_pos1", hexbits(pos1[0]))
    print("CASE graph_pos2", hexbits(pos2[0]))

    # --- empty/degenerate room ---
    empty = {"perception": [], "prediction": [], "vibe": {}}
    compute_vibe(empty)
    print("CASE empty_strength", hexbits(empty["vibe"]["strength"]))
    print("CASE empty_pos0", hexbits(empty["vibe"]["position"][0]))

    print("DEMO OK — grand-pattern-mojo end-to-end receipt")


if __name__ == "__main__":
    sys.exit(main())
