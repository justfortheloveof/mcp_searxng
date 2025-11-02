test:
	uv run pytest
.PHONY: test

test-show-output:
	uv run pytest --capture=no
.PHONY: test-show-output

clean:
	rm -v test_mcp_searxng.py.log
.PHONY: clean
