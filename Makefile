UV_CACHE_DIR := /goinfre/$(USER)/.cache/uv
UV_PROJECT_ENVIRONMENT := /goinfre/$(USER)/.venv
HF_HOME := /goinfre/$(USER)/.cache/huggingface
TMPDIR := /goinfre/$(USER)/tmp

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT
export HF_HOME
export TMPDIR

.PHONY: install run debug lint lint-strict clean

install:
	@mkdir -p $(UV_CACHE_DIR) $(UV_PROJECT_ENVIRONMENT) $(HF_HOME) $(TMPDIR)
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 . --extend-exclude=llm_sdk,.venv
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --extend-exclude=llm_sdk,.venv
	uv run mypy . --strict

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache data/output
