#!/usr/bin/env python3
"""Grand Pattern — bit-for-bit parity checker.

Normalizes hex tokens (Mojo's hex() emits 0x-prefixed variable-width; Python
side matches) and compares every token of every CASE line:

    python3 python/parity_check.py outputs/demo_mojo.txt outputs/demo_python.txt

Exit 0 = full parity, 1 = divergence.
"""

import sys


def canon(tok: str) -> str:
    if tok.startswith("0x") or tok.startswith("0X"):
        return format(int(tok, 16), "016x")
    if tok.lstrip("-").isdigit():
        return "int:" + str(int(tok))
    return tok


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: parity_check.py <mojo_output> <python_output>")
        return 2
    mojo_lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    py_lines = [l.strip() for l in open(sys.argv[2]) if l.strip()]

    if len(mojo_lines) != len(py_lines):
        print(f"LINE COUNT DIVERGENCE: mojo={len(mojo_lines)} python={len(py_lines)}")
        return 1

    total = 0
    bad = 0
    for ml, pl in zip(mojo_lines, py_lines):
        total += 1
        mt = [canon(t) for t in ml.split()]
        pt = [canon(t) for t in pl.split()]
        if mt != pt:
            bad += 1
            print(f"DIVERGE  mojo: {ml}")
            print(f"DIVERGE python: {pl}")
        else:
            print(f"OK       {ml.split()[1] if len(ml.split()) > 1 else ml}")

    print("")
    print(f"Parity: {total - bad}/{total} lines bit-identical")
    if bad:
        print("PARITY FAILED")
        return 1
    print("PARITY OK — bit-for-bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
