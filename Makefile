.PHONY: install run debug lint lint-strict clean

install:
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