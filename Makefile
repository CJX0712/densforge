# DensForge developer tasks.
#
# `make check` runs exactly what CI runs, in the same order, so a local green and
# a CI green mean the same thing.

PYTHON ?= python
RUFF   ?= ruff
OUT    ?= benchmark.json
DATASETS ?=
ifeq ($(strip $(DATASETS)),)
DATASET_ARGS :=
else
DATASET_ARGS := --datasets $(DATASETS)
endif

.PHONY: help install dev lint fmt format test cov smoke demo bench check clean docker all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with the loose constraints
	$(PYTHON) -m pip install -e .

dev:  ## Pinned dependencies plus test tooling
	$(PYTHON) -m pip install -r requirements.lock.txt
	$(PYTHON) -m pip install pytest pytest-cov ruff

# `ruff format` rewrites line breaks, so it must run after pyproject.toml exists.
# The order here is deliberate and matches .github/workflows/ci.yml.
lint:  ## ruff check
	$(RUFF) check .

fmt: format  ## Alias for format

format:  ## ruff format
	$(RUFF) format .

# The double invocation is intentional: `format` rewrites, `format --check`
# verifies. A single `format` would always pass and prove nothing.
lint-strict:  ## ruff check + ruff format --check, as CI runs it
	$(RUFF) check .
	$(RUFF) format --check .

test:  ## Run the test suite
	$(PYTHON) -m pytest -q -W ignore::UserWarning

# The 80% floor is a floor, not a target. The modules carrying the project's
# claims sit far above it; the floor catches a new module arriving untested.
cov:  ## Coverage with the 80% floor enforced
	$(PYTHON) -m pytest -q -W ignore::UserWarning --cov=densforge --cov-report=term-missing --cov-fail-under=80

smoke:  ## The only check that exercises the CLI, pipeline and artefact round-trip
	$(PYTHON) examples/run_demo.py --quick

demo:  ## Full-scale demo
	$(PYTHON) examples/run_demo.py

bench:  ## Full benchmark -> $(OUT)
	$(PYTHON) -m densforge.cli bench --out $(OUT) --seeds 3 $(DATASET_ARGS)

# CI's order, exactly. Running them in a different order locally hides
# interactions: `format` rewrites files that `check` then has to re-verify.
check: lint-strict test cov smoke  ## What CI runs

docker:  ## Build the CPU-only image
	docker build -t densforge:local .

all: check  ## Alias for check

clean:  ## Remove build and test artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -f benchmark.json benchmark.json.tmp
