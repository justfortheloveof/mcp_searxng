all: clean format test-all
.PHONY: all

clean:
	rm -vrf ./_*.log ./.coverage* ./htmlcov
.PHONY: clean

clean-all: clean
	rm -vrf ./__pycache__ ./.pytest_cache/ ./.venv
.PHONY: clean-all

format:
	uv run black .

lint:
	uv run black --check --diff  --color .
	uv run basedpyright .

test:
	uv run pytest -n auto
.PHONY: test

test-all: lint test
.PHONY: test-all

test-show-output:
	uv run pytest -n auto --capture=no
.PHONY: test-show-output
