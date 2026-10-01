# Grand Pattern Fibonacci Dual-Direction Architecture - Mojo
#
# Toolchain on this box (Mojo 1.2.0-dev): exports required if mojo is not on
# PATH — see docs/USERMANUAL.md. `-I .` is REQUIRED for all targets: imports
# resolve `from src.types import *` relative to -I, not cwd/main-file dir.

MOJO ?= mojo
MOJOFLAGS := -I .

.PHONY: all test build demo parity clean

all: test

build:
	$(MOJO) build $(MOJOFLAGS) tests/test_gp.mojo -o test_gp

test:
	$(MOJO) run -D ASSERT=all $(MOJOFLAGS) tests/test_gp.mojo

demo:
	$(MOJO) run $(MOJOFLAGS) examples/demo.mojo

parity: 
	mkdir -p outputs
	$(MOJO) run $(MOJOFLAGS) examples/demo.mojo > outputs/demo_mojo.txt
	python3 python/oracle_gp.py > outputs/demo_python.txt
	python3 python/parity_check.py outputs/demo_mojo.txt outputs/demo_python.txt

clean:
	rm -f test_gp
