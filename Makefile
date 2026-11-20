PYTHON ?= python
SHELL := /bin/sh
export PYTHONPATH := src
export MPLBACKEND := Agg

.PHONY: all test demo gen train eval figures clean check

all: test eval

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) -m capacity_tunnel.cli demo --payload 64

gen:
	$(PYTHON) -m capacity_tunnel.cli gen --out results

train: gen
	$(PYTHON) -m capacity_tunnel.cli train --data results/dataset.npz --out results

eval:
	$(PYTHON) -m capacity_tunnel.cli eval --out results
	$(PYTHON) scripts/make_figures.py --results results/results.json --flows results/flow_scores.npz --out results

figures:
	$(PYTHON) scripts/make_figures.py --results results/results.json --flows results/flow_scores.npz --out results

check:
	$(PYTHON) scripts/check_repro.py

clean:
	rm -rf results/results.json results/model.json results/metrics.json results/dataset.json
	rm -rf results/dataset.npz results/flow_scores.npz results/fig_*.png
	rm -rf src/capacity_tunnel/__pycache__ tests/__pycache__ scripts/__pycache__
