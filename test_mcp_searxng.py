# TODO: validate CLI args are parsed properly
import copy
import json
import os
import subprocess
from collections.abc import AsyncGenerator
from typing import TypedDict, cast

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from fastmcp.exceptions import ToolError

from mcp_searxng import FitSearXNGResponse, FitSearXNGResponseWithHint


class SearXNGServerConfig(TypedDict):
    transport: str
    command: str
    args: list[str]
    cwd: str
    env: dict[str, str]


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
                    "--log-to", "_test.log",
                    "--log-level", "DEBUG",
                    "--no-ssl-verify",
                    # fmt: on
                ],
                "cwd": os.getcwd(),
                "env": {  # added for pytest coverage to be picked up by subprocesses
                    "COVERAGE_PROCESS_START": os.path.join(os.getcwd(), "pyproject.toml"),
                },
            }
        }
    }


@pytest_asyncio.fixture(scope="function")
async def mcp_client(
    mcp_server_config: dict[str, dict[str, dict[str, str | list[str]]]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    """Keep the MCP client connected for all tests."""
    client = Client(mcp_server_config)
    async with client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def mcp_client_with_hint_and_custom_tool(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    """Keep the MCP client connected for tests that need hints enabled."""
    config = copy.deepcopy(mcp_server_config)
    config["mcpServers"]["searxng"]["args"].extend(["--include-hint", "--web-fetch-tool-name", "testing"])

    client = Client(config)
    async with client:
        yield client


@pytest.mark.asyncio
async def test_arg_override_env_missing_server_url_arg(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
):
    config = copy.deepcopy(mcp_server_config)
    server_config = cast(SearXNGServerConfig, config["mcpServers"]["searxng"])  # pyright: ignore[reportUnnecessaryCast]
    # remove `[... , "--server-url", "URL", ...]` from args
    idx = server_config["args"].index("--server-url")
    del server_config["args"][idx:idx + 2]  # fmt: skip

    # Extract the command and args from config
    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    # Run the subprocess and capture stderr
    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    # Expect non-zero exit code and check stderr contains the message
    assert result.returncode != 0
    assert "--override-env requires --server-url URL to be provided" in result.stderr


@pytest.mark.asyncio
async def test_log_level_arg_missing_log_to_arg(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
):
    config = copy.deepcopy(mcp_server_config)
    server_config = cast(SearXNGServerConfig, config["mcpServers"]["searxng"])  # pyright: ignore[reportUnnecessaryCast]
    # remove `[... , "--log-to", "FILE_PATH", ...]` from args
    idx = server_config["args"].index("--log-to")
    del server_config["args"][idx:idx + 2]  # fmt: skip

    # Extract the command and args from config
    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    # Run the subprocess and capture stderr
    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    # Expect non-zero exit code and check stderr contains the message
    assert result.returncode != 0
    assert "--log-to is required when --log-level is provided" in result.stderr


@pytest.mark.asyncio
async def test_engines_arg_with_spaces(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = cast(SearXNGServerConfig, config["mcpServers"]["searxng"])  # pyright: ignore[reportUnnecessaryCast]
    server_config["args"].extend(["--engines", "t e s t"])

    # Extract the command and args from config
    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    # Run the subprocess and capture stderr
    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    # Expect non-zero exit code and check stderr contains the message
    assert result.returncode != 0
    assert "--engines must not contain spaces" in result.stderr


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
async def test_call_tool_searxng_web_search_with_empty_query(mcp_client: Client[MCPConfigTransport]) -> None:
    expected_exc_str = "The 'query' field cannot be empty"
    with pytest.raises(ToolError, match=f"^{expected_exc_str}$"):
        _ = await mcp_client.call_tool("searxng_web_search", {"query": "   "})


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search(mcp_client: Client[MCPConfigTransport]) -> None:
    searxng_web_search_results = await mcp_client.call_tool("searxng_web_search", {"query": "testing 1 2 1 2"})
    print("\ncall_tool 'searxng_web_search' output:")
    print(json.dumps(searxng_web_search_results.structured_content, ensure_ascii=False, indent=4))
    assert searxng_web_search_results.structured_content is not None, "no structured_content in tool response"
    fit_search_response = FitSearXNGResponse.model_validate(searxng_web_search_results.structured_content["result"])
    assert (
        len(fit_search_response.results) > 0
    ), "searxng_web_search mcp tool response should contain more than 0 search results"
    assert not hasattr(
        fit_search_response, "number_of_results"
    ), "FitSearXNGResponse should not have unfit attribute(s)"
    assert not hasattr(
        fit_search_response.results[0], "category"
    ), "FitSearXNGResult should not have unfit attribute(s)"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_hint(
    mcp_client_with_hint_and_custom_tool: Client[MCPConfigTransport],
) -> None:
    searxng_web_search_results = await mcp_client_with_hint_and_custom_tool.call_tool(
        "searxng_web_search", {"query": "testing 3 4 3 4"}
    )
    print("\ncall_tool 'searxng_web_search' with hint output:")
    print(json.dumps(searxng_web_search_results.structured_content, ensure_ascii=False, indent=4))
    assert searxng_web_search_results.structured_content is not None, "no structured_content in tool response"
    fit_search_response_w_hint = FitSearXNGResponseWithHint.model_validate(
        searxng_web_search_results.structured_content["result"]
    )
    assert fit_search_response_w_hint.hint is not None, "'hint' attribute of tool response cannot be None"
    assert (
        "testing tool" in fit_search_response_w_hint.hint
    ), "custom tool name not found in 'hint' attribute of tool response"
    assert (
        len(fit_search_response_w_hint.results) > 0
    ), "searxng_web_search mcp tool response should contain more than 0 search results"
    assert not hasattr(
        fit_search_response_w_hint, "number_of_results"
    ), "FitSearXNGResponseWithHint should not have unfit attribute(s)"
    assert not hasattr(
        fit_search_response_w_hint.results[0], "category"
    ), "FitSearXNGResult should not have unfit attribute(s)"
