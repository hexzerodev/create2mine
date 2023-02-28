# ---------------------------------------------------------------------------- #
#                                     setup                                    #
# ---------------------------------------------------------------------------- #
.PHONY: install
install:
	cargo install cargo-modules

# ---------------------------------------------------------------------------- #
#                                     lint                                     #
# ---------------------------------------------------------------------------- #
.PHONY: lint
lint:
	cargo clippy

.PHONY: format
format:
	cargo fmt

# ---------------------------------------------------------------------------- #
#                                     misc                                     #
# ---------------------------------------------------------------------------- #
.PHONY: modules
modules:
	cargo modules generate tree --with-types --bin create2mine
	cargo modules generate tree --with-types --lib

# ---------------------------------------------------------------------------- #
#                                      run                                     #
# ---------------------------------------------------------------------------- #
.PHONY: run
run:
	cargo run \
		0xf6b72a0854388fbb9e0f4882001491Fe94Aa81C3 \
		0xf6b72a0854388fbb9e0f4882001491Fe94Aa81C3 \
		0xf6b72a0854388fbb9e0f4882001491Fe94Aa81C391Fe94Aa81C391Fe94Aa81C3

.PHONY: run.release
run.release:
	cargo run --release a b c
