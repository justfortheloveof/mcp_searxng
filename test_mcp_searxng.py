import base64
import copy
import json
import os
import ssl
import subprocess
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn, TypedDict, cast

import httpx
import pytest
import pytest_asyncio
import trustme
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import MCPConfigTransport
from fastmcp.exceptions import ToolError
from pytest import TempPathFactory
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from mcp_searxng import FitSearXNGResponse, FitSearXNGResponseWithHint, SearXNGResponse

MOCK_FIT_SEARXNG_RESPONSE = FitSearXNGResponse.model_validate(
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


MOCK_SEARXNG_RESPONSE = SearXNGResponse.model_validate(
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
        "unresponsive_engines": [],
    }
).model_dump()


class SearXNGServerConfig(TypedDict):
    transport: str
    command: str
    args: list[str]
    cwd: str
    env: dict[str, str]


@dataclass
class AuthConfig:
    auth_type: str
    args: list[str]
    header: dict[str, str]


@dataclass
class ErrorConfig:
    name: str
    status: int | None
    data: str | None
    handler: Callable[[Request], NoReturn] | None
    expected_pattern: str


@dataclass
class ArgValidationConfig:
    args: list[str]
    expected_stderr: str


AUTH_CONFIGS: list[AuthConfig] = [
    AuthConfig(
        auth_type="basic",
        args=["--no-ssl-verify", "--auth-type", "basic", "--auth-username", "testuser", "--auth-password", "testpass"],
        header={"Authorization": "Basic " + base64.b64encode(b"testuser:testpass").decode()},
    ),
    AuthConfig(
        auth_type="bearer",
        args=["--no-ssl-verify", "--auth-type", "bearer", "--auth-token", "testtoken"],
        header={"Authorization": "Bearer testtoken"},
    ),
    AuthConfig(
        auth_type="api_key",
        args=["--no-ssl-verify", "--auth-type", "api_key", "--auth-api-key", "testkey"],
        header={"X-API-Key": "testkey"},
    ),
]

ERROR_CONFIGS: list[ErrorConfig] = [
    ErrorConfig(
        name="401_error",
        status=401,
        data='{"error": "Unauthorized"}',
        handler=None,
        expected_pattern="Authentication to SearXNG server failed: Invalid credentials provided",
    ),
    ErrorConfig(
        name="403_error",
        status=403,
        data='{"error": "Forbidden"}',
        handler=None,
        expected_pattern=(
            "Access to SearXNG server forbidden: Authentication may be required or credentials lack permission"
        ),
    ),
    ErrorConfig(
        name="500_error",
        status=500,
        data='{"error": "Internal Server Error"}',
        handler=None,
        expected_pattern=r"SearXNG request failed with status 500: .*",
    ),
    ErrorConfig(
        name="connection_error",
        status=None,
        data=None,
        handler=lambda request: (_ for _ in ()).throw(httpx.ConnectError("Connection failed")),
        expected_pattern=r"SearXNG request failed with status 500: .*",
    ),
    ErrorConfig(
        name="invalid_json_error",
        status=200,
        data="invalid json content",
        handler=None,
        expected_pattern=r"SearXNG request failed: .*",
    ),
]

ARG_VALIDATION_CONFIGS: list[ArgValidationConfig] = [
    ArgValidationConfig(
        args=[  # --override-env with missing --server-url
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--override-env",
        ],
        expected_stderr="--override-env requires --server-url=URL",
    ),
    ArgValidationConfig(
        args=[  # --log-level missing --log-to
            "run",
            "./mcp_searxng.py",
            "--log-level",
            "DEBUG",
        ],
        expected_stderr="--log-to is required when --log-level is provided",
    ),
    ArgValidationConfig(
        args=[  # invalid engines with spaces
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--engines",
            "t e s t",
        ],
        expected_stderr="--engines must not contain spaces",
    ),
    ArgValidationConfig(
        args=[  # invalid log path (path/parent doesn't exist)
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "/non_existent/directory/log_file.log",
        ],
        expected_stderr="The directory for the log file '/non_existent/directory/log_file.log' does not exist",
    ),
    ArgValidationConfig(
        args=[  # invalid log path (dir)
            "run",
            "./mcp_searxng.py",
            "--log-to",
            ".",
        ],
        expected_stderr="The log file path must be a file, a symlink to a file or a fifo: ",
    ),
    ArgValidationConfig(
        args=[  # non-existent CA file
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--ssl-ca-file",
            "/non_existent/ssl_ca.pem",
        ],
        expected_stderr="The SSL CA path does not exist: /non_existent/ssl_ca.pem",
    ),
    ArgValidationConfig(
        args=[  # invalid CA file (directory)
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--ssl-ca-file",
            ".",
        ],
        expected_stderr="The SSL CA path must be a file or a symlink to a file: ",
    ),
    ArgValidationConfig(
        args=[  # --no-ssl-verify, but --ssl-ca-file present
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--no-ssl-verify",
            "--ssl-ca-file",
            "README.md",
        ],
        expected_stderr="--no-ssl-verify cannot be used with auth or when an SSL CA file is provided",
    ),
    ArgValidationConfig(
        args=[  # HTTP URL with auth
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--server-url",
            "http://test",
            "--auth-type",
            "basic",
        ],
        expected_stderr="Authentication requires HTTPS for security. Please use an HTTPS URL",
    ),
    ArgValidationConfig(
        args=[  # missing username/password
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--server-url",
            "https://test",
            "--override-env",
            "--auth-type",
            "basic",
        ],
        expected_stderr="--auth-type=basic requires --auth-username and --auth-password",
    ),
    ArgValidationConfig(
        args=[  # missing token
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--server-url",
            "https://test",
            "--override-env",
            "--auth-type",
            "bearer",
        ],
        expected_stderr="--auth-type=bearer requires --auth-token",
    ),
    ArgValidationConfig(
        args=[  # missing API key
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--server-url",
            "https://test",
            "--override-env",
            "--auth-type",
            "api_key",
        ],
        expected_stderr="--auth-type=api_key requires --auth-api-key",
    ),
    ArgValidationConfig(
        args=[  # only one engine with --engines-rotate
            "run",
            "./mcp_searxng.py",
            "--log-to",
            "_test.log",
            "--log-level",
            "DEBUG",
            "--engines",
            "bogus",
            "--engines-rotate",
        ],
        expected_stderr="--engines-rotate requires at least two engines to be provided with --engines",
    ),
]


def run_subprocess_and_assert_error(server_config: SearXNGServerConfig, expected_stderr: str) -> None:
    """Helper to run subprocess and assert error for arg validation tests"""
    cmd = [server_config["command"]] + server_config["args"]
    env = server_config.get("env", {})
    result = subprocess.run(cmd, cwd=server_config["cwd"], env={**os.environ, **env}, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected_stderr in result.stderr


def assert_searxng_tool_response(results: CallToolResult, expected_query: str = "test query") -> FitSearXNGResponse:
    """Helper to assert common tool response structure"""
    assert results.structured_content is not None
    fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
    assert len(fit_response.results) > 0
    assert fit_response.query == expected_query
    return fit_response


@pytest.fixture(scope="session")
def httpserver_ssl_context() -> ssl.SSLContext:
    """Create a self-signed SSL certificate for HTTPS testing"""
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost", "127.0.0.1", "::1")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_cert.configure_cert(ctx)  # pyright: ignore[reportUnknownMemberType]

    return ctx


@pytest.fixture(scope="function")
def httpserver_ssl(httpserver_ssl_context: ssl.SSLContext) -> Generator[HTTPServer, None, None]:
    """Create an HTTPS test server to support HTTPS auth testing"""
    server = HTTPServer(ssl_context=httpserver_ssl_context)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="function")
def httpserver_ssl_with_ca_file(tmp_path_factory: TempPathFactory) -> Generator[tuple[str, HTTPServer], None, None]:
    ca = trustme.CA()
    ca_cert_path: Path = tmp_path_factory.mktemp("ssl") / "ca.pem"
    ca.cert_pem.write_to_path(ca_cert_path)

    server_cert = ca.issue_cert("localhost", "127.0.0.1", "::1")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_cert.configure_cert(ctx)  # pyright: ignore[reportUnknownMemberType]

    server = HTTPServer(ssl_context=ctx)
    server.start()
    yield str(ca_cert_path), server
    server.stop()
    if server.is_running():
        server.clear()
    try:
        ca_cert_path.unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture(scope="session")
def mcp_server_config() -> dict[str, dict[str, SearXNGServerConfig]]:
    """Setup and return server config after checking SEARXNG_URL"""
    searxng_url = os.getenv("SEARXNG_URL", None)
    if searxng_url is None:
        # TODO: support mocked tests (with optional prod test)
        pytest.fail("SEARXNG_URL environment variable must be set for tests")

    # TODO: remove this crap
    return {
        "mcpServers": {
            "searxng": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "./mcp_searxng.py",
                    "--override-env",
                    "--server-url",
                    searxng_url,
                    "--log-to",
                    "_test.log",
                    "--log-level",
                    "DEBUG",
                ],
                "cwd": os.getcwd(),
                "env": {  # added for pytest coverage to be picked up by subprocesses
                    "COVERAGE_PROCESS_START": os.path.join(os.getcwd(), "pyproject.toml"),
                },
            }
        }
    }


@pytest.fixture(scope="function", params=AUTH_CONFIGS)
def mcp_server_config_auth(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
    request: pytest.FixtureRequest,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    """Parametrized fixture for authentication configurations"""
    param: AuthConfig = cast(AuthConfig, request.param)
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url
    server_config["args"].extend(param.args)
    server_config["args"].append("--no-ssl-verify")

    httpserver_ssl.expect_request(
        "/search",
        method="GET",
        header_value_matcher=param.header,  # pyright: ignore[reportArgumentType]
    ).respond_with_json(MOCK_FIT_SEARXNG_RESPONSE)

    return config


@pytest.fixture(scope="function")
def mcp_server_config_ssl_ca_file(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl_with_ca_file: tuple[str, HTTPServer],
) -> dict[str, dict[str, SearXNGServerConfig]]:
    ca_file_path, httpserver = httpserver_ssl_with_ca_file
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url
    server_config["args"].extend(["--ssl-ca-file", ca_file_path])

    httpserver.expect_request("/search", method="GET").respond_with_json(MOCK_FIT_SEARXNG_RESPONSE)

    return config


@pytest.fixture(scope="function", params=ERROR_CONFIGS)
def mcp_server_config_error(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
    request: pytest.FixtureRequest,
) -> tuple[dict[str, dict[str, SearXNGServerConfig]], str]:
    param: ErrorConfig = cast(ErrorConfig, request.param)
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url

    if param.handler is not None:
        httpserver_ssl.expect_request("/search", method="GET").respond_with_handler(param.handler)
    else:
        assert param.data is not None, "param.data must be non-None when handler is None"
        assert param.status is not None, "param.status must be non-None when handler is None"
        httpserver_ssl.expect_request("/search", method="GET").respond_with_data(
            param.data,
            status=param.status,
            content_type="application/json",
        )

    return config, param.expected_pattern


@pytest.fixture(scope="function")
def mcp_server_config_rotation_success(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url
    server_config["args"].extend(["--engines-rotate", "--engines", "failing_engine,succeeding_engine"])

    httpserver_ssl.expect_request("/search", method="GET").respond_with_json(MOCK_FIT_SEARXNG_RESPONSE)
    httpserver_ssl.expect_request("/search", method="GET").respond_with_json(
        {"query": "test", "results": [], "unresponsive_engines": [["failing_engine", "error"]]}
    )

    return config


@pytest.fixture(scope="function")
def mcp_server_config_rotation_fallback(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url
    server_config["args"].extend(["--engines-rotate", "--engines", "fail1,fail2"])

    httpserver_ssl.expect_request("/search").respond_with_handler(
        lambda request: Response(
            json.dumps(
                MOCK_SEARXNG_RESPONSE
                if "engines=" in str(request.url) and "engines=fail" not in str(request.url)
                else (
                    {"query": "test", "results": [], "unresponsive_engines": [["fail1", "error"]]}
                    if "engines=fail1" in str(request.url)
                    else (
                        {
                            "query": "test",
                            "results": [],
                            "unresponsive_engines": [["fail1", "error"], ["fail2", "error"]],
                        }
                        if "engines=fail2" in str(request.url)
                        else {"error": "no match"}
                    )
                )
            ),
            mimetype="application/json",
        )
    )

    return config


@pytest.fixture(scope="function")
def mcp_server_config_rotation_empty_engines(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url
    server_config["args"].extend(["--engines-rotate", "--engines", ",,,"])

    return config


@pytest.fixture(scope="function")
def mcp_server_config_all_engines_unresponsive(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    httpserver_ssl: HTTPServer,
) -> dict[str, dict[str, SearXNGServerConfig]]:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]
    server_url_idx = server_config["args"].index("--server-url") + 1
    https_url = httpserver_ssl.url_for("/").replace("http://", "https://")

    server_config["args"][server_url_idx] = https_url

    httpserver_ssl.expect_request("/search", method="GET").respond_with_json(
        {
            "query": "test query",
            "results": [],
            "unresponsive_engines": [["duckduckgo", "error"], ["brave", "error"], ["startpage", "error"]],
        }
    )

    return config


@pytest_asyncio.fixture(scope="function")
async def mcp_client(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    client = Client(mcp_server_config)
    async with client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def mcp_client_with_hint_and_custom_tool(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
) -> AsyncGenerator[Client[MCPConfigTransport], None]:
    config = copy.deepcopy(mcp_server_config)
    config["mcpServers"]["searxng"]["args"].extend(["--include-hint", "--web-fetch-tool-name", "testing"])

    client = Client(config)
    async with client:
        yield client


def _arg_validation_config_id(config: ArgValidationConfig) -> str:
    return config.expected_stderr


@pytest.mark.parametrize("validation_config", ARG_VALIDATION_CONFIGS, ids=_arg_validation_config_id)
@pytest.mark.asyncio
async def test_arg_validation(
    mcp_server_config: dict[str, dict[str, SearXNGServerConfig]],
    validation_config: ArgValidationConfig,
) -> None:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    server_config["args"] = validation_config.args

    run_subprocess_and_assert_error(server_config, validation_config.expected_stderr)


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_auth(
    mcp_server_config_auth: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_auth)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test"})
        _ = assert_searxng_tool_response(results)


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_ssl_ca_file(
    mcp_server_config_ssl_ca_file: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_ssl_ca_file)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test"})
        _ = assert_searxng_tool_response(results)


@pytest.mark.asyncio
async def test_searxng_url_not_set(mcp_server_config: dict[str, dict[str, SearXNGServerConfig]]) -> None:
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # remove --server-url and --override-env arg and clear env to simulate no URL provided
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
    config = copy.deepcopy(mcp_server_config)
    server_config = config["mcpServers"]["searxng"]

    # replace --server-url with an invalid URL (no scheme)
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
    assert list_tools_result[0].name == "searxng_web_search", (
        f"Expected tool name 'searxng_web_search', got {list_tools_result[0].name}"
    )


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
    assert len(fit_search_response.results) > 0, (
        "searxng_web_search mcp tool response should contain more than 0 search results"
    )
    assert not hasattr(fit_search_response, "number_of_results"), (
        "FitSearXNGResponse should not have unfit attribute(s)"
    )
    assert not hasattr(fit_search_response.results[0], "category"), (
        "FitSearXNGResult should not have unfit attribute(s)"
    )


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
    assert "testing tool" in fit_search_response_w_hint.hint, (
        "custom tool name not found in 'hint' attribute of tool response"
    )
    assert len(fit_search_response_w_hint.results) > 0, (
        "searxng_web_search mcp tool response should contain more than 0 search results"
    )
    assert not hasattr(fit_search_response_w_hint, "number_of_results"), (
        "FitSearXNGResponseWithHint should not have unfit attribute(s)"
    )
    assert not hasattr(fit_search_response_w_hint.results[0], "category"), (
        "FitSearXNGResult should not have unfit attribute(s)"
    )


@pytest.mark.asyncio
async def test_engines_rotate_with_empty_engines_list(
    mcp_server_config_rotation_empty_engines: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    client = Client(mcp_server_config_rotation_empty_engines)
    async with client:
        with pytest.raises(ToolError, match="No engines configured for rotation"):
            _ = await client.call_tool("searxng_web_search", {"query": "test query"})


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_engine_rotation_success(
    mcp_server_config_rotation_success: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    config = copy.deepcopy(mcp_server_config_rotation_success)
    server_config = config["mcpServers"]["searxng"]

    server_config["args"].append("--no-ssl-verify")

    client = Client(config)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test query"})
        assert results.structured_content is not None, "no structured_content in tool response"
        fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
        assert len(fit_response.results) > 0
        assert fit_response.query == "test query"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_with_engine_rotation_fallback(
    mcp_server_config_rotation_fallback: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    config = copy.deepcopy(mcp_server_config_rotation_fallback)
    server_config = config["mcpServers"]["searxng"]

    server_config["args"].append("--no-ssl-verify")

    client = Client(config)
    async with client:
        results = await client.call_tool("searxng_web_search", {"query": "test query"})
        assert results.structured_content is not None, "no structured_content in tool response"
        fit_response = FitSearXNGResponse.model_validate(results.structured_content["result"])
        assert len(fit_response.results) > 0
        assert fit_response.query == "test query"


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_all_engines_unresponsive(
    mcp_server_config_all_engines_unresponsive: dict[str, dict[str, SearXNGServerConfig]],
) -> None:
    expected_msg = (
        r"It seems like all requested SearXNG engines were unresponsive: "
        r"\[\['duckduckgo', 'error'\], \['brave', 'error'\], \['startpage', 'error'\]\]"
    )
    config = copy.deepcopy(mcp_server_config_all_engines_unresponsive)
    server_config = config["mcpServers"]["searxng"]

    server_config["args"].append("--no-ssl-verify")

    client = Client(config)
    async with client:
        with pytest.raises(ToolError, match=f"^{expected_msg}$"):
            _ = await client.call_tool("searxng_web_search", {"query": "test query"})


@pytest.mark.asyncio
async def test_call_tool_searxng_web_search_error(
    mcp_server_config_error: tuple[dict[str, dict[str, SearXNGServerConfig]], str],
) -> None:
    config, expected_pattern = mcp_server_config_error
    server_config = config["mcpServers"]["searxng"]

    server_config["args"].append("--no-ssl-verify")

    client = Client(config)
    async with client:
        with pytest.raises(ToolError, match=expected_pattern):
            _ = await client.call_tool("searxng_web_search", {"query": "test query"})
