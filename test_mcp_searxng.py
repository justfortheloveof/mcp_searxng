import asyncio
import json
import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport

from mcp_searxng import FitSearXNGResponse, FitSearXNGResponseWithHint


@pytest_asyncio.fixture(scope="function")
async def mcp_client(
    mcp_server_config: dict[str, dict[str, dict[str, str | list[str]]]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    """Keep the MCP client connected for all tests."""
    client = Client(mcp_server_config)
    async with client:
        await asyncio.sleep(0.5)  # fastmcp is slow to start
        yield client


@pytest_asyncio.fixture(scope="function")
async def mcp_client_with_hint_and_custom_tool(
    mcp_server_config: dict[str, dict[str, dict[str, str | list[str]]]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    """Keep the MCP client connected for tests that need hints enabled."""
    config = mcp_server_config.copy()
    if isinstance(config["mcpServers"]["searxng"]["args"], list):
        config["mcpServers"]["searxng"]["args"].extend(["--include-hint", "--web-fetch-tool-name", "testing"])
    else:
        pytest.fail("config['mcpServers']['searxng']['args'] must be a list")
    client = Client(config)
    async with client:
        await asyncio.sleep(0.5)  # fastmcp is slow to start
        yield client


@pytest.fixture(scope="session")
def mcp_server_config() -> dict[str, dict[str, dict[str, str | list[str] | dict[str, str]]]]:
    """Setup and return server config after checking SEARXNG_URL."""
    searxng_url = os.getenv("SEARXNG_URL", None)
    if searxng_url is None:
        pytest.fail("SEARXNG_URL environment variable must be set for tests")

    return {
        "mcpServers": {
            "searxng": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    # fmt: off
                    "run", "./mcp_searxng.py",
                    "--override-env",
                    "--server-url", searxng_url,
                    "--log-to", "test_mcp_searxng.py.log",
                    # fmt: on
                ],
                "cwd": os.getcwd(),
                "env": {  # added for pytest coverage to be picked up by subprocesses
                    "COVERAGE_PROCESS_START": os.path.join(os.getcwd(), "pyproject.toml"),
                },
            }
        }
    }


@pytest.mark.asyncio
async def test_server_ping(mcp_client: Client[MCPConfigTransport]) -> None:
    ping_result = await mcp_client.ping()
    assert ping_result is True, f"Expected ping to return True, got {ping_result}"


@pytest.mark.asyncio
async def test_list_tools(mcp_client: Client[MCPConfigTransport]) -> None:
    list_tools_result = await mcp_client.list_tools()
    print(f"\nlist_tools output:{list_tools_result}")
    assert len(list_tools_result) == 1, f"Expected 1 tool, got {len(list_tools_result)}"
    assert (
        list_tools_result[0].name == "searxng_web_search"
    ), f"Expected tool name 'searxng_web_search', got {list_tools_result[0].name}"


@pytest.mark.asyncio
async def test_call_tool_read(mcp_client: Client[MCPConfigTransport]) -> None:
    searxng_web_search_results = await mcp_client.call_tool("searxng_web_search", {"query": "testing 1 2 1 2"})
    print("\ncall_tool 'searxng_web_search' output:")
    print(json.dumps(searxng_web_search_results.structured_content, ensure_ascii=False, indent=4))
    assert searxng_web_search_results.structured_content is not None, "no structured_content in tool response"
    response = FitSearXNGResponse.model_validate(searxng_web_search_results.structured_content["result"])
    assert len(response.results) > 0, "searxng_web_search mcp tool response contains 0 search results"


@pytest.mark.asyncio
async def test_call_tool_with_hint(mcp_client_with_hint_and_custom_tool: Client[MCPConfigTransport]) -> None:
    searxng_web_search_results = await mcp_client_with_hint_and_custom_tool.call_tool(
        "searxng_web_search", {"query": "testing 3 4 3 4"}
    )
    print("\ncall_tool 'searxng_web_search' with hint output:")
    print(json.dumps(searxng_web_search_results.structured_content, ensure_ascii=False, indent=4))
    assert searxng_web_search_results.structured_content is not None, "no structured_content in tool response"
    response = FitSearXNGResponseWithHint.model_validate(searxng_web_search_results.structured_content["result"])
    assert response.hint is not None, "'hint' attribute of tool response cannot be None"
    assert "testing tool" in response.hint, "custom tool name not found in 'hint' attribute of tool response"
    assert len(response.results) > 0, "searxng_web_search mcp tool response contains 0 search results"
