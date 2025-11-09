clean:
	rm -vrf ./_*.log ./.coverage*
.PHONY: clean

clean-all: clean
	rm -vrf ./__pycache__ ./.pytest_cache/ ./.venv
.PHONY: clean-all

format:
	uv run black .

lint:
	uv run basedpyright .

test:
	uv sync --dev
	uv run black --check --diff --color .
	uv run basedpyright .
	uv run pytest -n auto
.PHONY: test

test-show-output:
	uv sync --dev
	uv run black --check --diff  --color .
	uv run basedpyright .
	uv run pytest -n auto --capture=no
.PHONY: test-show-output
