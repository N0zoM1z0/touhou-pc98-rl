.PHONY: setup native paper test

setup:
	uv sync

native:
	uv sync --reinstall-package touhou-pc98-rl

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

test:
	cargo test --lib
	uv run python -m unittest discover -s tests -v
	uv run python -m compileall -q pc98rl scripts tests
	bash -n scripts/patch_th05.sh scripts/prepare_hdi.sh
