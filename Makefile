test:
	uv run pytest
.PHONY: test

test-show-output:
	uv run pytest --capture=no
.PHONY: test-show-output

clean:
	rm -vrf _*.log .coverage
.PHONY: clean
