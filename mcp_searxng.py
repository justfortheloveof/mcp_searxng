import argparse
import json
import logging
import os
from typing import Annotated, Any, ClassVar, Literal, cast
from pydantic import BaseModel, ConfigDict, Field
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP


# TODO: use pydantic-settings for better settings management?
class MCPSearXNGConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    # TODO: review types, need None?
    arg_engines: str = "duckduckgo,brave,startpage"
    arg_include_hint: bool = False
    arg_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    arg_log_to: str | None = None
    arg_ssl_verify: bool = False
    arg_override_env: bool = False
    arg_server_url: str | None = None
    arg_web_fetch_tool_name: str = "webfetch"
    env_searxng_url: str = Field(default_factory=lambda: os.getenv("SEARXNG_URL", ""))
    searxng_url: str = ""


# WARN: Global vars...
config = MCPSearXNGConfig.model_construct()
log = logging.getLogger(__name__)
mcp = FastMCP("mcp_searxng")


class SearXNGResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    url: str | None = Field(None, description="The URL of the search result")
    title: str | None = Field(None, description="The title of the search result")
    content: str | None = Field(None, description="The content snippet of the search result")
    engine: str | None = Field(None, description="The search engine that provided this result")
    score: float | None = Field(None, description="The relevance score of the result")
    category: str | None = None
    parsed_url: list[str] | None = None
    template: str | None = None
    positions: list[int] | None = None
    priority: Literal["", "high", "low"] | None = None
    thumbnail: str | None = None
    publishedDate: str | None = None
    pretty_url: str | None = None
    img_src: str | None = None
    iframe_src: str | None = None
    audio_src: str | None = None
    pubdate: str | None = None
    length: str | None = None
    views: str | None = None
    author: str | None = None
    metadata: str | None = None
    engines: list[str] | None = None
    open_group: bool | None = None
    close_group: bool | None = None


class FitSearXNGResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    url: str | None = Field(None, description="The URL of the search result")
    title: str | None = Field(None, description="The title of the search result")
    content: str | None = Field(None, description="The content snippet of the search result")
    engine: str | None = Field(None, description="The search engine that provided this result")
    score: float | None = Field(None, description="The relevance score of the result")


class SearXNGResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    query: str
    results: list[SearXNGResult]
    number_of_results: int | None = None
    answers: list[dict[str, Any]] | None = None  # pyright: ignore[reportExplicitAny]
    corrections: list[str] | None = None
    infoboxes: list[dict[str, Any]] | None = None  # pyright: ignore[reportExplicitAny]
    suggestions: list[str] | None = None
    unresponsive_engines: list[list[str]] | None = None


class FitSearXNGResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    query: str = Field(description="The search query string")
    results: list[FitSearXNGResult] = Field(description="List of search results")


class FitSearXNGResponseWithHint(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    query: str
    results: list[FitSearXNGResult]
    hint: str


class SearXNGSearchParams(BaseModel):
    # NOTE: hard coded params, except:
    # `q`: provided in tool call
    # `engines`: passed as CLI arg
    # notice we only ever return the first page of results
    # https://docs.searxng.org/dev/search_api.html
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    q: str
    language: str = Field(default="all")
    pageno: int = Field(default=1, gt=0)
    safesearch: int = Field(default=0, ge=0, le=2)
    format: str = Field(default="json", pattern="^(json|csv|rss)$")
    engines: str = Field(default=config.arg_engines)


def parse_args_update_config() -> None:
    parser = argparse.ArgumentParser(description="MCP server to search and read web URLs using SearXNG")

    _ = parser.add_argument(
        "--server-url",
        "-s",
        type=str,
        metavar="URL",
        help=(
            "SearXNG server URL."
            "The value in the SEARXNG_URL environment variable takes precedence, unless --override-env is also used."
        ),
    )
    _ = parser.add_argument(
        "--override-env",
        "-o",
        action="store_true",
        default=False,
        help="Override environment variables with command line arguments.",
    )
    _ = parser.add_argument(
        "--engines",
        type=str,
        default=config.arg_engines,
        help=(
            f"Comma-separated list of SearXNG engines to use for searches (default: {config.arg_engines})."
            "An empty string enables all engines."
        ),
    )
    _ = parser.add_argument(
        "--include-hint",
        action="store_true",
        default=config.arg_include_hint,
        help="Include a hint in search results suggesting to use the web fetch tool.",
    )
    _ = parser.add_argument(
        "--web-fetch-tool-name",
        type=str,
        default=config.arg_web_fetch_tool_name,
        help=f"Name of the web fetch tool to reference in LLM hint (default: {config.arg_web_fetch_tool_name}).",
    )
    _ = parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        default=config.arg_ssl_verify,
        help="Disable SSL certificate verification - unsafe",
    )
    _ = parser.add_argument(
        "--log-to",
        type=str,
        metavar="LOG_FILE_PATH",
        help="Enable logging to file - required when --log-level is provided.",
    )
    _ = parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help=f"Set logging level for file output (default: {config.arg_log_level}).",
    )

    args = parser.parse_args()

    # TODO: clean up all these casts
    engines = cast(str, args.engines)
    log_level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None, args.log_level)
    log_to = cast(str | None, args.log_to)
    override_env = cast(bool, args.override_env)
    server_url = cast(str | None, args.server_url)

    if override_env and not server_url:
        parser.error("`--override-env` requires `--server-url URL` to be provided")

    if log_level and not log_to:
        parser.error("`--log-to LOG_FILE_PATH` is required when `--log-level` is provided")

    engines = engines.strip()
    if " " in engines:
        parser.error("--engines cannot contain spaces")

    server_url = server_url.rstrip("/") if server_url else server_url

    config.arg_engines = engines
    config.arg_include_hint = cast(bool, args.include_hint)
    config.arg_log_level = log_level or config.arg_log_level
    config.arg_log_to = log_to
    config.arg_ssl_verify = cast(bool, not args.no_ssl_verify)
    config.arg_override_env = override_env
    config.arg_server_url = server_url
    config.arg_web_fetch_tool_name = cast(str, args.web_fetch_tool_name)


def setup_logger(config: MCPSearXNGConfig) -> None:
    if config.arg_log_to:
        logging.basicConfig(
            level=logging.getLevelNamesMapping().get(config.arg_log_level),
            format="[%(asctime)s %(levelname)s] %(message)s",
            # TODO: validate this is a path?
            handlers=[logging.FileHandler(config.arg_log_to)],
        )


def setup_mcp_server_config(config: MCPSearXNGConfig) -> None:
    """Set SearXNG server URL based on CLI arguments."""
    # TODO: Support auth
    if (not config.env_searxng_url or config.arg_override_env) and config.arg_server_url:
        config.searxng_url = config.arg_server_url
        log.debug(f"SearXNG server URL set from command line argument: {config.searxng_url}")
    else:
        config.searxng_url = config.env_searxng_url
        log.debug(f"SearXNG server URL set from environment variable: {config.searxng_url}")

    _validate_config(config)

    log.debug(f"Include hint: {config.arg_include_hint}, Web fetch tool name: {config.arg_web_fetch_tool_name}")
    log.debug(f"Search engines set to: {config.arg_engines}")


def _validate_config(config: MCPSearXNGConfig) -> None:
    """Check that the SearXNG server URL is set and valid, raise otherwise"""
    if not config.searxng_url:
        msg = (
            "SearXNG server URL is not set. "
            "Please provide a valid URL via the SEARXNG_URL environment variable or command line argument."
        )
        log.critical(msg)
        raise ValueError(msg)

    parsed_url = urlparse(config.searxng_url)

    if not all([parsed_url.scheme, parsed_url.netloc]):
        msg = f"Invalid SearXNG URL '{config.searxng_url}'. Please provide a valid URL."
        log.critical(msg)
        raise ValueError(msg)


async def search(search_params: SearXNGSearchParams) -> FitSearXNGResponse:
    url = f"{config.searxng_url}/search"
    timeout = 10.0  # perhaps should be a CLI arg?

    log.info(f"requesting SearXNG search at {url} with params: {search_params}, timeout: {timeout}")

    async with httpx.AsyncClient(verify=config.arg_ssl_verify) as client:
        response = await client.get(url, timeout=timeout, params=search_params.model_dump(exclude_none=True))

    _ = response.raise_for_status()
    log.info(f"Response received from SearXNG with status code: {response.status_code}")

    search_response = SearXNGResponse.model_validate(response.json())

    validate_search_response(search_params, search_response)
    log.debug(
        f"Validated Raw SearXNGResponse:\n{json.dumps(search_response.model_dump(), ensure_ascii=False, indent=2)}"
    )
    fit_search_response = FitSearXNGResponse.model_validate(search_response.model_dump())

    return fit_search_response


def validate_search_response(search_params: SearXNGSearchParams, search_response: SearXNGResponse) -> None:
    """Check if the search response is valid, meaning at least 1 search engine was responsive"""
    if search_response.unresponsive_engines:
        log.warning(f"Unresponsive SearXNG engine(s): {search_response.unresponsive_engines}")

        if not search_response.results and (
            len(search_response.unresponsive_engines) == len(search_params.engines.split(","))
        ):
            msg = (
                f"It seems like all requested SearXNG engines were unresponsive: {search_response.unresponsive_engines}"
            )
            log.error(msg)
            raise RuntimeError(msg)


@mcp.tool
async def searxng_web_search(
    query: Annotated[str, "The web search query string"],
) -> FitSearXNGResponse | FitSearXNGResponseWithHint:
    """Search the web"""
    log.info(f"searxng_web_search tool called with query: '{query}'")

    if query.strip() == "":
        log.error("Search query is an empty string")
        # TODO: use ToolError to control message to LLM
        raise ValueError("Search query cannot be empty")

    search_params = SearXNGSearchParams(q=query, engines=config.arg_engines)

    search_response = await search(search_params)

    if config.arg_include_hint:
        search_response_dict = search_response.model_dump(exclude_none=True)
        search_response_dict["hint"] = (
            "These are the web search results for your query. Each result is a web page and "
            f"you can access its whole content using the url value with the {config.arg_web_fetch_tool_name} tool"
        )
        search_response = FitSearXNGResponseWithHint.model_validate(search_response_dict)

    formatted_response = json.dumps(search_response.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
    log.info(f"searxng_web_search tool call completed with FitSearXNGResponse:\n{formatted_response}")

    return search_response


def main() -> int:
    parse_args_update_config()
    setup_logger(config)

    log.info("Starting MCP SearXNG server")

    setup_mcp_server_config(config)
    mcp.run(show_banner=False)

    log.info("MCP SearXNG server stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
