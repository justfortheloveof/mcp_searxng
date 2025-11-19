import logging
import runpy
import sys
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import respx
from fastmcp.exceptions import ToolError
from httpx import ConnectError, Request, Response

import mcp_searxng
from mcp_searxng import (
    FitSearXNGResponse,
    FitSearXNGResponseWithHint,
    MCPSearXNGArgs,
    MCPSearXNGConfig,
    MCPSearXNGEnvVars,
    SearXNGResponse,
    SearXNGResult,
    searxng_web_search,
)

MOCK_URL = "https://mock-searxng.local"
MOCK_SEARCH_RESULT = SearXNGResponse(
    query="test query",
    results=[
        SearXNGResult(
            url="https://test.co",
            title="TestTitle",
            content="Test content",
            engine="test",
            score=1.0,
        )
    ],
    unresponsive_engines=[],
).model_dump()


@pytest.fixture(autouse=True)
def mock_sys_argv() -> Generator[None, None, None]:
    """patch sys.argv to prevent pydantic-settings from trying to parse pytest args."""
    with patch.object(sys, "argv", ["mcp_searxng"]):
        yield


@pytest.fixture
def mock_env_vars() -> MCPSearXNGEnvVars:
    return MCPSearXNGEnvVars(URL=MOCK_URL)


@pytest.fixture
def mock_args() -> MCPSearXNGArgs:
    return MCPSearXNGArgs()


@pytest.fixture
def mock_config(mock_env_vars: MCPSearXNGEnvVars, mock_args: MCPSearXNGArgs) -> Generator[MCPSearXNGConfig, None, None]:
    config = MCPSearXNGConfig(args=mock_args, env=mock_env_vars)
    # change mcp_searxng `config` module level variable
    mcp_searxng.config = config
    yield config


@pytest.fixture
def searxng_api(respx_mock: respx.MockRouter) -> respx.Route:
    """Mock the SearXNG API endpoint."""
    return respx_mock.get(f"{MOCK_URL}/search")


# test main and entry point
def test_main() -> None:
    """Test main() function directly."""
    with (
        patch("fastmcp.FastMCP.run") as mock_run,
        patch("sys.argv", ["mcp_searxng", "--server-url", MOCK_URL]),
        patch("mcp_searxng.setup_logger") as mock_logger,
    ):
        with patch.dict("os.environ", {"SEARXNG_URL": ""}):
            ret = mcp_searxng.main()
            assert ret == 0
            mock_run.assert_called_once()
            mock_logger.assert_called_once()


def test_entrypoint_runpy() -> None:
    with (
        patch("fastmcp.FastMCP.run") as mock_run,
        patch("sys.argv", ["mcp_searxng", "--server-url", MOCK_URL]),
        patch("mcp_searxng.setup_logger"),
    ):
        with pytest.raises(SystemExit) as excinfo:
            _ = runpy.run_path("mcp_searxng.py", run_name="__main__")

        assert excinfo.value.code == 0
        mock_run.assert_called_once()


# test cli arguments validation
def test_arg_validation_override_env_missing_url() -> None:
    with pytest.raises(SystemExit, match="--override-env requires --server-url=URL"):
        _ = MCPSearXNGArgs(override_env=True, server_url=None)


def test_arg_validation_log_level_missing_log_to() -> None:
    with pytest.raises(SystemExit, match="--log-to is required when --log-level is provided"):
        _ = MCPSearXNGArgs(log_level="DEBUG", log_to=None)


def test_arg_validation_engines_spaces() -> None:
    with pytest.raises(SystemExit, match="--engines must not contain spaces"):
        _ = MCPSearXNGArgs(engines="foo bar")


def test_arg_validation_rotation_not_enough_engines() -> None:
    with pytest.raises(SystemExit, match="--engines-rotate requires at least two engines"):
        _ = MCPSearXNGArgs(engines_rotate=True, engines="onlyone")


def test_validation_log_to_parent_missing() -> None:
    with pytest.raises(SystemExit, match="The directory for the log file .* does not exist"):
        _ = MCPSearXNGArgs(log_to="/nonexistent/dir/file.log")


def test_validation_log_to_is_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="The log file path must be a file"):
        _ = MCPSearXNGArgs(log_to=str(tmp_path))


def test_validation_ssl_ca_missing() -> None:
    with pytest.raises(SystemExit, match="The SSL CA path does not exist"):
        _ = MCPSearXNGArgs(ssl_ca_file="/nonexistent/ca.pem")


def test_validation_ssl_ca_is_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="The SSL CA path must be a file"):
        _ = MCPSearXNGArgs(ssl_ca_file=str(tmp_path))


# test config logic
def test_config_validation_no_url() -> None:
    env = MCPSearXNGEnvVars(URL="")
    args = MCPSearXNGArgs(server_url=None)
    with pytest.raises(SystemExit, match="SearXNG server URL is not set"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_validation_invalid_url_scheme() -> None:
    env = MCPSearXNGEnvVars(URL="ftp://invalid.com")
    args = MCPSearXNGArgs()
    with pytest.raises(SystemExit, match="Please provide a valid URL"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_validation_auth_requires_https() -> None:
    env = MCPSearXNGEnvVars(URL="http://fail")
    args = MCPSearXNGArgs(auth_type="basic")
    with pytest.raises(SystemExit, match="Authentication requires HTTPS"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_validation_no_ssl_verify_but_cert(tmp_path: Path) -> None:
    ca_file = tmp_path / "ca.pem"
    ca_file.touch()
    env = MCPSearXNGEnvVars(URL=MOCK_URL)
    args = MCPSearXNGArgs(ssl_verify=False, ssl_ca_file=str(ca_file))
    with pytest.raises(SystemExit, match="--no-ssl-verify cannot be used with auth or when an SSL CA file is provided"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_validation_no_ssl_verify_but_auth() -> None:
    env = MCPSearXNGEnvVars(URL=MOCK_URL)
    args = MCPSearXNGArgs(ssl_verify=False, auth_type="basic")
    with pytest.raises(SystemExit, match="--no-ssl-verify cannot be used with auth or when an SSL CA file is provided"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_server_url_from_args() -> None:
    env = MCPSearXNGEnvVars(URL="")
    args = MCPSearXNGArgs(server_url=MOCK_URL)
    config = MCPSearXNGConfig(args=args, env=env)
    assert config.searxng_url == MOCK_URL


def test_config_override_env() -> None:
    env = MCPSearXNGEnvVars(URL="https://loose")
    args = MCPSearXNGArgs(server_url="https://win", override_env=True)
    config = MCPSearXNGConfig(args=args, env=env)
    assert config.searxng_url == "https://win"


def test_config_auth_missing_basic_args() -> None:
    env = MCPSearXNGEnvVars(URL=MOCK_URL)
    args = MCPSearXNGArgs(auth_type="basic", auth_username="u", auth_password=None)
    with pytest.raises(SystemExit, match="requires --auth-username and --auth-password"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_auth_missing_bearer_token() -> None:
    env = MCPSearXNGEnvVars(URL=MOCK_URL)
    args = MCPSearXNGArgs(auth_type="bearer", auth_token=None)
    with pytest.raises(SystemExit, match="requires --auth-token"):
        _ = MCPSearXNGConfig(args=args, env=env)


def test_config_auth_missing_api_key() -> None:
    env = MCPSearXNGEnvVars(URL=MOCK_URL)
    args = MCPSearXNGArgs(auth_type="api_key", auth_api_key=None)
    with pytest.raises(SystemExit, match="requires --auth-api-key"):
        _ = MCPSearXNGConfig(args=args, env=env)


# test helpers
def test_setup_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    args = MCPSearXNGArgs(log_to=str(log_file), log_level="DEBUG")
    config = MCPSearXNGConfig(args=args, env=MCPSearXNGEnvVars(URL=MOCK_URL))
    with patch("logging.basicConfig") as mock_logging:
        mcp_searxng.setup_logger(config)
        mock_logging.assert_called_once()
        kwargs = mock_logging.call_args.kwargs
        assert isinstance(cast(list[logging.Handler], kwargs["handlers"])[0], logging.FileHandler)
        assert len(cast(list[logging.Handler], kwargs["handlers"])) == 1


def test_redact_config_secrets() -> None:
    args = MCPSearXNGArgs(auth_password="secret_pass", auth_token="secret_token", auth_api_key="secret_key")
    config = MCPSearXNGConfig(args=args, env=MCPSearXNGEnvVars(URL=MOCK_URL))
    redacted = mcp_searxng.redact_config_secrets(config)
    assert redacted.args.auth_password == "***password***"
    assert redacted.args.auth_token == "***token***"
    assert redacted.args.auth_api_key == "***apikey***"
    assert config.args.auth_password == "secret_pass"


# test mcp tool
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_search_success(searxng_api: respx.Route) -> None:
    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    result = cast(FitSearXNGResponse, await searxng_web_search.fn(query="test query"))

    assert isinstance(result, FitSearXNGResponse)
    assert result.query == "test query"
    assert len(result.results) == 1
    assert result.results[0].title == "TestTitle"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_search_empty_query() -> None:
    with pytest.raises(ToolError, match="The 'query' field cannot be empty"):
        await searxng_web_search.fn(query="   ")


@pytest.mark.asyncio
async def test_search_with_hint(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.include_hint = True
    mock_config.args.web_fetch_tool_name = "my_fetcher"

    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    result = cast(FitSearXNGResponseWithHint, await searxng_web_search.fn(query="test"))

    assert isinstance(result, FitSearXNGResponseWithHint)
    assert "my_fetcher" in result.hint
    assert result.hint.startswith("These are the web search results")


@pytest.mark.asyncio
async def test_search_with_ssl_ca_file(mock_config: MCPSearXNGConfig, searxng_api: respx.Route, tmp_path: Path) -> None:
    ca_file = tmp_path / "ca.pem"
    ca_file.touch()
    mock_config.args.ssl_ca_file = str(ca_file)

    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    with patch("ssl.create_default_context") as mock_ssl:
        context = MagicMock()
        mock_ssl.return_value = context

        await searxng_web_search.fn(query="test")

        mock_ssl.assert_called_with(cafile=str(ca_file))


# test auth
@pytest.mark.asyncio
async def test_auth_basic(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.auth_type = "basic"
    mock_config.args.auth_username = "user"
    mock_config.args.auth_password = "pass"

    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    await searxng_web_search.fn(query="test")

    request = searxng_api.calls.last.request
    assert request.headers["Authorization"] == "Basic dXNlcjpwYXNz"  # user:pass base64


@pytest.mark.asyncio
async def test_auth_bearer(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.auth_type = "bearer"
    mock_config.args.auth_token = "secret_token"

    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    await searxng_web_search.fn(query="test")

    request = searxng_api.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret_token"


@pytest.mark.asyncio
async def test_auth_api_key(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.auth_type = "api_key"
    mock_config.args.auth_api_key = "mykey"
    mock_config.args.auth_api_key_header = "X-Custom-Key"

    searxng_api.return_value = Response(200, json=MOCK_SEARCH_RESULT)

    await searxng_web_search.fn(query="test")

    request = searxng_api.calls.last.request
    assert request.headers["X-Custom-Key"] == "mykey"


# test searxng server response (http codes)
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_error_401(searxng_api: respx.Route) -> None:
    searxng_api.return_value = Response(401)

    with pytest.raises(ToolError, match="Authentication to SearXNG server failed"):
        await searxng_web_search.fn(query="test")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_error_403(searxng_api: respx.Route) -> None:
    searxng_api.return_value = Response(403)

    with pytest.raises(ToolError, match="Access to SearXNG server forbidden"):
        await searxng_web_search.fn(query="test")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_error_500_generic(searxng_api: respx.Route) -> None:
    searxng_api.return_value = Response(500)

    with pytest.raises(ToolError, match="SearXNG request failed with status 500"):
        await searxng_web_search.fn(query="test")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_error_connection(searxng_api: respx.Route) -> None:
    searxng_api.side_effect = ConnectError("Connection failed")

    with pytest.raises(ToolError, match="SearXNG request failed"):
        await searxng_web_search.fn(query="test")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_error_all_engines_unresponsive(searxng_api: respx.Route) -> None:
    resp = MOCK_SEARCH_RESULT.copy()
    resp["results"] = []
    resp["unresponsive_engines"] = [["duckduckgo", "error"], ["brave", "error"]]  # type: ignore

    searxng_api.return_value = Response(200, json=resp)

    with pytest.raises(ToolError, match="It seems like all requested SearXNG engines were unresponsive"):
        await searxng_web_search.fn(query="test")


# test engine rotation
@pytest.mark.asyncio
async def test_rotation_success(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.engines_rotate = True
    mock_config.args.engines = "engine1,engine2"

    # first request: engine1 fails
    def side_effect(request: Request) -> Response:
        params: dict[str, str] = dict(request.url.params)
        if params.get("engines") == "engine1":
            return Response(200, json={"query": "test", "results": [], "unresponsive_engines": [["engine1", "error"]]})
        if params.get("engines") == "engine2":
            return Response(200, json=MOCK_SEARCH_RESULT)
        return Response(500)

    searxng_api.side_effect = side_effect

    result = cast(FitSearXNGResponse, await searxng_web_search.fn(query="test"))

    assert len(result.results) == 1
    assert searxng_api.call_count == 2


@pytest.mark.asyncio
async def test_rotation_all_fail_fallback(mock_config: MCPSearXNGConfig, searxng_api: respx.Route) -> None:
    mock_config.args.engines_rotate = True
    mock_config.args.engines = "e1,e2"

    # e1 and e2 fail, fallback (no engines param) succeeds
    def side_effect(request: Request) -> Response:
        params: dict[str, str] = dict(request.url.params)
        engines = params.get("engines", "")
        if engines in ["e1", "e2"]:
            return Response(200, json={"query": "test", "results": [], "unresponsive_engines": [[engines, "error"]]})
        if engines == "":  # fallback
            return Response(200, json=MOCK_SEARCH_RESULT)
        return Response(500)

    searxng_api.side_effect = side_effect

    result = cast(FitSearXNGResponse, await searxng_web_search.fn(query="test"))

    assert len(result.results) == 1
    assert searxng_api.call_count == 3  # e1, e2, fallback


@pytest.mark.asyncio
async def test_rotation_no_engines_configured(mock_config: MCPSearXNGConfig) -> None:
    mock_config.args.engines_rotate = True
    mock_config.args.engines = ",,,"

    with pytest.raises(ToolError, match="No engines configured for rotation"):
        await searxng_web_search.fn(query="test")
