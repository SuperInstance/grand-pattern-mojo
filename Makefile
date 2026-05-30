# Grand Pattern Fibonacci Dual-Direction Architecture - Mojo
# Placeholder Makefile

.PHONY: all test clean

all:
	mojo build tests/test_gp.mojo -o test_gp

test:
	mojo run tests/test_gp.mojo

clean:
	rm -f test_gp
