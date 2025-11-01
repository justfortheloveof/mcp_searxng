import asyncio
import json
import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from mcp.types import Tool

mcp_server_config = {}


@pytest_asyncio.fixture(scope="function")
async def mcp_client() -> AsyncGenerator[Client[MCPConfigTransport], None]:
    """Keep the MCP client connected for all tests."""
    client = Client(mcp_server_config)
    async with client:
        await asyncio.sleep(0.5)  # fastmcp is slow to start
        yield client


def test_env_var() -> None:
    """Must be run first as it sets the URL for the server connection used in the other tests"""
    searxng_url = os.getenv("SEARXNG_URL", None)
    assert searxng_url is not None, "SEARXNG_URL environment variable must be set for tests"

    global mcp_server_config
    mcp_server_config = {
        "mcpServers": {
            "searxng": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "./mcp_searxng.py", "--override-env", "--server-url", searxng_url],
                "cwd": os.getcwd(),
            }
        }
    }


@pytest.mark.asyncio
async def test_server_ping(mcp_client: Client[MCPConfigTransport]) -> None:
    ping_result = await mcp_client.ping()
    assert ping_result is True


@pytest.mark.asyncio
async def test_list_tools(mcp_client: Client[MCPConfigTransport]) -> None:
    list_tools_result = await mcp_client.list_tools()
    print(f"\nlist_tools output:{list_tools_result}")
    assert len(list_tools_result) == 1
    assert isinstance(list_tools_result[0], Tool)
    assert list_tools_result[0].name == "searxng_web_search"


@pytest.mark.asyncio
async def test_call_tool_read(mcp_client: Client[MCPConfigTransport]) -> None:
    searxng_web_search_results = await mcp_client.call_tool("searxng_web_search", {"query": "testing 1 2 1 2"})
    print("\ncall_tool 'searxng_web_search' output:")
    print(json.dumps(searxng_web_search_results.structured_content, ensure_ascii=False, indent=4))
    assert searxng_web_search_results.structured_content is not None
    assert "hint" in searxng_web_search_results.structured_content
    assert "query" in searxng_web_search_results.structured_content
    assert "results" in searxng_web_search_results.structured_content
    assert isinstance(searxng_web_search_results.structured_content["results"], list)
    assert len(searxng_web_search_results.structured_content["results"]) > 0
    for result in searxng_web_search_results.structured_content["results"]:
        assert isinstance(result, dict)
        assert "title" in result
        assert "content" in result
        assert "url" in result
