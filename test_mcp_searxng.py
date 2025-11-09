import base64
import copy
import json
import os
import ssl
import subprocess
from collections.abc import AsyncGenerator, Generator
from typing import TypedDict

import pytest
import pytest_asyncio
import trustme
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from fastmcp.exceptions import ToolError
from pytest_httpserver import HTTPServer

from mcp_searxng import FitSearXNGResponse, FitSearXNGResponseWithHint


MOCK_SEARXNG_RESPONSE = FitSearXNGResponse.model_validate(
    {
        "query": "test query",
        "results": [
            {
                "url": "https://test.co",
                "title": "TestTitle",
                "content": "Test content",
                "engine": "test",
                "score": 1.0,
            }
        ],
    }
).model_dump()


class SearXNGServerConfig(TypedDict):
    transport: str
    command: str
    args: list[str]
    cwd: str
    env: dict[str, str]


@pytest.fixture(scope="session")
def httpserver_ssl_context() -> ssl.SSLContext:
    """Create a self-signed SSL certificate for HTTPS testing."""
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost", "127.0.0.1", "::1")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_cert.configure_cert(ctx)  # pyright: ignore[reportUnknownMemberType]

    return ctx


@pytest.fixture(scope="function")
def httpserver_ssl(httpserver_ssl_context: ssl.SSLContext) -> Generator[HTTPServer, None, None]:
    """Create an HTTPS test server to support HTTPS auth requirements."""
    server = HTTPServer(ssl_context=httpserver_ssl_context)
    server.start()
    yield server
    server.stop()
    if server.is_running():
        server.clear()


@pytest.fixture(scope="session")
def mcp_server_config() -> dict[str, dict[str, SearXNGServerConfig]]:
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


@pytest.fixture(scope="function")
def mcp_server_config_basic_auth(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    """MCP config for basic auth tests, reusing and modifying mcp_server_config."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")
    server_config["args"][server_url_idx] = https_url

    server_config["args"].extend(["--auth-type", "basic", "--auth-username", "testuser", "--auth-password", "testpass"])

    expected_auth = "Basic " + base64.b64encode(b"testuser:testpass").decode()
    httpserver_ssl.expect_request(
        "/search",
        method="GET",
        header_value_matcher={"Authorization": expected_auth},  # pyright: ignore[reportArgumentType]
    ).respond_with_json(MOCK_SEARXNG_RESPONSE)

    return config


@pytest.fixture(scope="function")
def mcp_server_config_bearer_auth(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    """MCP config for bearer auth tests, reusing and modifying mcp_server_config."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")
    server_config["args"][server_url_idx] = https_url

    server_config["args"].extend(["--auth-type", "bearer", "--auth-token", "testtoken"])

    httpserver_ssl.expect_request(
        "/search",
        method="GET",
        header_value_matcher={"Authorization": "Bearer testtoken"},  # pyright: ignore[reportArgumentType]
    ).respond_with_json(MOCK_SEARXNG_RESPONSE)

    return config


@pytest.fixture(scope="function")
def mcp_server_config_api_key_auth(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    """MCP config for API key auth tests, reusing and modifying mcp_server_config."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")
    server_config["args"][server_url_idx] = https_url

    server_config["args"].extend(["--auth-type", "api_key", "--auth-api-key", "testkey"])

    httpserver_ssl.expect_request(
        "/search", method="GET", header_value_matcher={"X-API-Key": "testkey"}  # pyright: ignore[reportArgumentType]
    ).respond_with_json(MOCK_SEARXNG_RESPONSE)

    return config


@pytest_asyncio.fixture(scope="function")
async def mcp_client(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
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
    server_config = config["mcpServers"]["searxng"]
    # remove `[... , "--server-url", "URL", ...]` from args
    idx = server_config["args"].index("--server-url")
    del server_config["args"][idx:idx + 2]  # fmt: skip

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--override-env requires --server-url=URL" in result.stderr


@pytest.mark.asyncio
async def test_log_level_arg_missing_log_to_arg(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    # remove `[... , "--log-to", "FILE_PATH", ...]` from args
    idx = server_config["args"].index("--log-to")
    del server_config["args"][idx:idx + 2]  # fmt: skip

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--log-to is required when --log-level is provided" in result.stderr


@pytest.mark.asyncio
async def test_engines_arg_with_spaces(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--engines", "t e s t"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--engines must not contain spaces" in result.stderr


@pytest.mark.asyncio
async def test_log_to_arg_nonexistent_parent_directory(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    log_to_idx = server_config["args"].index("--log-to") + 1
    bogus_dir = "/non_existent/directory/log_file.log"
    server_config["args"][log_to_idx] = bogus_dir

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert result.stderr == f"The directory for the log file '{bogus_dir}' does not exist\n"


@pytest.mark.asyncio
async def test_log_to_arg_points_to_directory(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    log_to_idx = server_config["args"].index("--log-to") + 1
    server_config["args"][log_to_idx] = "."

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "The log file path must be a file, a symlink to a file or a fifo: " in result.stderr


@pytest.mark.asyncio
async def test_ssl_ca_file_nonexistent_path(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    if "--no-ssl-verify" in server_config["args"]:
        server_config["args"].remove("--no-ssl-verify")
    server_config["args"].extend(["--ssl-ca-file", "/non_existent/ssl_ca.pem"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert result.stderr == "The SSL CA path does not exist: /non_existent/ssl_ca.pem\n"


@pytest.mark.asyncio
async def test_ssl_ca_file_points_to_directory(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    if "--no-ssl-verify" in server_config["args"]:
        server_config["args"].remove("--no-ssl-verify")
    server_config["args"].extend(["--ssl-ca-file", "."])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "The SSL CA path must be a file or a symlink to a file: " in result.stderr


@pytest.mark.asyncio
async def test_ssl_verify_conflict_with_ssl_ca_file(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--ssl-ca-file", "pyproject.toml"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert result.stderr == "--no-ssl-verify cannot be used when --ssl-ca-file is provided\n"


@pytest.mark.asyncio
async def test_auth_requires_https(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]) -> None:
    """Test that the CLI exits when auth is enabled but URL is not HTTPS."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # Replace --server-url with HTTP URL and add basic auth
    server_url_idx = server_config["args"].index("--server-url") + 1
    server_config["args"][server_url_idx] = "http://test"
    server_config["args"].extend(["--auth-type", "basic", "--auth-username", "test", "--auth-password", "test"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Authentication requires HTTPS for security. Please use an HTTPS URL" in result.stderr


@pytest.mark.asyncio
async def test_auth_type_basic_missing_credentials(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--auth-type", "basic"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--auth-type=basic requires --auth-username and --auth-password" in result.stderr


@pytest.mark.asyncio
async def test_auth_type_bearer_missing_token(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--auth-type", "bearer"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--auth-type=bearer requires --auth-token" in result.stderr


@pytest.mark.asyncio
async def test_auth_type_api_key_missing_key(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--auth-type", "api_key"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--auth-type=api_key requires --auth-api-key" in result.stderr


@pytest.mark.asyncio
async def test_engines_rotate_requires_at_least_two_engines(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
):
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_config["args"].extend(["--engines-rotate", "--engines", "bogus"])

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert result.stderr == "--engines-rotate requires at least two engines to be provided with --engines\n"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_basic_auth(
    mcp_server_config_basic_auth: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_basic_auth)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test"})
        assert results.structured_content is not None, "no structured_content in tool response"
        fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
        assert len(fit_response.results) > 0
        assert fit_response.query == "test query"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_bearer_auth(
    mcp_server_config_bearer_auth: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_bearer_auth)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test"})
        assert results.structured_content is not None, "no structured_content in tool response"
        fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
        assert len(fit_response.results) > 0
        assert fit_response.query == "test query"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_api_key_auth(
    mcp_server_config_api_key_auth: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_api_key_auth)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test"})
        assert results.structured_content is not None, "no structured_content in tool response"
        fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
        assert len(fit_response.results) > 0
        assert fit_response.query == "test query"


@pytest.mark.asyncio
async def test_searxng_url_not_set(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]) -> None:
    """Test that the CLI exits when no SearXNG URL is provided via env var or --server-url."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # Remove --server-url and --override-env arg and clear env to simulate no URL provided
    idx = server_config["args"].index("--server-url")
    del server_config["args"][idx: idx + 2]  # fmt: skip
    idx = server_config["args"].index("--override-env")
    del server_config["args"][idx: idx + 1]  # fmt: skip
    server_config["env"] = {}  # Clear env vars, including SEARXNG_URL
    _ = os.environ.pop("SEARXNG_URL")

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "SearXNG server URL is not set" in result.stderr


@pytest.mark.asyncio
async def test_invalid_searxng_url_missing_scheme(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]) -> None:
    """Test that the CLI exits for an invalid URL missing a scheme."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # Replace --server-url with an invalid URL (no scheme)
    server_url_idx = server_config["args"].index("--server-url") + 1
    server_config["args"][server_url_idx] = "test"

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Invalid SearXNG URL 'test'" in result.stderr
    assert "Please provide a valid URL (must start with http:// or https://)" in result.stderr


@pytest.mark.asyncio
async def test_invalid_searxng_url_unsupported_scheme(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    """Test that the CLI exits for an invalid URL with an unsupported scheme."""
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # Replace --server-url with an invalid URL (unsupported scheme)
    server_url_idx = server_config["args"].index("--server-url") + 1
    server_config["args"][server_url_idx] = "ftp://test"

    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})

    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Invalid SearXNG URL 'ftp://test'" in result.stderr
    assert "Please provide a valid URL (must start with http:// or https://)" in result.stderr


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
