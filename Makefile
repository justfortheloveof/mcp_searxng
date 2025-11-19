all: clean format test-all
.PHONY: all

clean:
	rm -vrf ./_*.log ./.coverage* ./htmlcov
.PHONY: clean

clean-all: clean
	rm -vrf ./__pycache__ ./.pytest_cache/ ./.venv
.PHONY: clean-all

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run basedpyright .

test:
	uv run pytest -n auto
.PHONY: test

test-all: lint test
.PHONY: test-all

test-show-output:
	uv run pytest -n auto --capture=no
.PHONY: test-show-output
