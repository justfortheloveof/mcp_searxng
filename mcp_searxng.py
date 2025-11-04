import argparse
import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, cast
from pydantic import BaseModel, ConfigDict, Field
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP


log = logging.getLogger(__name__)

searxng_url = os.getenv("SEARXNG_URL", "")
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
    publishedDate: str | None = None  # ISO datetime string
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

    query: str = Field(description="The search query string")
    results: list[SearXNGResult] = Field(description="List of search results")
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

    query: str = Field(description="The search query string")
    results: list[FitSearXNGResult] = Field(description="List of search results")
    hint: str | None = None  # added by MCP server (us)


@dataclass
class CLI_Args:
    server_url: str | None = None
    override_env: bool = False
    include_hint: bool = False
    web_fetch_tool_name: str = "webfetch"
    log_level: str | None = None
    log_to: str | None = None
    engines: str = "duckduckgo,brave,startpage"


def parse_args() -> CLI_Args:
    default_args = CLI_Args()
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
        "--include-hint",
        action="store_true",
        default=default_args.include_hint,
        help="Include a hint in search results suggesting to use the web fetch tool.",
    )
    _ = parser.add_argument(
        "--web-fetch-tool-name",
        type=str,
        default=default_args.web_fetch_tool_name,
        help=f"Name of the web fetch tool to reference in LLM hint (default: {default_args.web_fetch_tool_name}).",
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
        help=f"Set logging level for file output (default: {default_args.log_level}).",
    )
    _ = parser.add_argument(
        "--engines",
        type=str,
        default=default_args.engines,
        help=(
            f"Comma-separated list of SearXNG engines to use for searches (default: {default_args.engines})."
            "An empty string enables all engines."
        ),
    )

    args = parser.parse_args()
    server_url = cast(str | None, args.server_url)
    override_env = cast(bool, args.override_env)
    include_hint = cast(bool, args.include_hint)
    web_fetch_tool_name = cast(str, args.web_fetch_tool_name)
    log_to = cast(str | None, args.log_to)
    log_level = cast(str | None, args.log_level)
    engines = cast(str, args.engines)

    if override_env and not server_url:
        parser.error("`--override-env` requires `--server-url URL` to be provided")

    if log_level and not log_to:
        parser.error("`--log-to LOG_FILE_PATH` is required when `--log-level` is provided")

    engines = engines.strip()
    if " " in engines:
        parser.error("--engines cannot contain spaces")

    server_url = server_url.rstrip("/") if server_url else server_url

    return CLI_Args(
        server_url=server_url,
        override_env=override_env,
        include_hint=include_hint,
        web_fetch_tool_name=web_fetch_tool_name,
        log_level=log_level,
        log_to=log_to,
        engines=engines,
    )


def setup_logger(args: CLI_Args) -> None:
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if args.log_level:
        logging.basicConfig(
            level=LOG_LEVELS[args.log_level],
            format="[%(asctime)s %(levelname)s] %(message)s",
            handlers=[logging.FileHandler(cast(str, args.log_to))],
        )


def set_mcp_vars(args: CLI_Args) -> None:
    """Set SearXNG server URL based on CLI arguments."""
    # TODO: Support auth
    global searxng_url  # NOTE: already set to either the value of env. var. "SEARXNG_URL" or "" at module level
    if (not searxng_url or args.override_env) and args.server_url:
        searxng_url = args.server_url
        log.debug(f"SearXNG server URL set from command line argument: {searxng_url}")
    else:
        log.debug(f"SearXNG server URL set from environment variable: {searxng_url}")

    global include_hint
    global web_fetch_tool_name
    include_hint = args.include_hint
    web_fetch_tool_name = args.web_fetch_tool_name
    log.debug(f"Include hint: {include_hint}, Web fetch tool name: {web_fetch_tool_name}")

    global engines
    engines = args.engines
    log.debug(f"Search engines set to: {engines}")


def validate_mcp_vars() -> None:
    """Check that the SearXNG server URL is set and valid, raise otherwise"""
    if not searxng_url:
        msg = (
            "SearXNG server URL is not set. "
            "Please provide a valid URL via the SEARXNG_URL environment variable or command line argument."
        )
        log.critical(msg)
        raise ValueError(msg)

    parsed_url = urlparse(searxng_url)

    if not all([parsed_url.scheme, parsed_url.netloc]):
        msg = f"Invalid SearXNG URL '{searxng_url}'. Please provide a valid URL."
        log.critical(msg)
        raise ValueError(msg)


async def search(search_params: dict[str, str | int]) -> FitSearXNGResponse:
    url = f"{searxng_url}/search"
    timeout = 10.0

    log.info(f"requesting SearXNG search at {url} with params: {search_params}, timeout: {timeout}")

    # TODO: verify should be a CLI arg, we should probably also support a custom cert (Zscaler)
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(url, timeout=timeout, params=search_params)

    _ = response.raise_for_status()
    log.info(f"Response received from SearXNG with status code: {response.status_code}")

    search_response = SearXNGResponse.model_validate(response.json())

    validate_search_response(search_params, search_response)
    log.debug(
        f"Validated Raw SearXNGResponse:\n{json.dumps(search_response.model_dump(), ensure_ascii=False, indent=2)}"
    )
    fit_search_response = FitSearXNGResponse.model_validate(search_response.model_dump())

    return fit_search_response


def validate_search_response(search_params: dict[str, str | int], search_response: SearXNGResponse) -> None:
    """Check if the search response is valid, meaning at least 1 search engine was responsive"""
    if search_response.unresponsive_engines:
        log.warning(f"Unresponsive SearXNG engine(s): {search_response.unresponsive_engines}")

        if not search_response.results and len(search_response.unresponsive_engines) == len(
            cast(str, search_params["engines"]).split(",")
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
    log.info(f"searxng_web_search tool called with query: {query}")

    if query.strip() == "":
        log.error("Search query is an empty string")
        raise ValueError("Search query cannot be empty")

    # NOTE: hard coded params - notice we only ever return the first page of results
    # https://docs.searxng.org/dev/search_api.html
    search_params = {
        "q": query,
        "language": "all",
        "pageno": 1,
        "safesearch": 0,
        "format": "json",
    }
    if engines:
        search_params["engines"] = engines

    try:
        search_response = await search(search_params)

        if include_hint:
            search_response = FitSearXNGResponseWithHint.model_validate(search_response.model_dump(exclude_none=True))
            search_response.hint = (
                "These are the web search results for your query. Each result is a web page and "
                f"you can access its whole content using the url value with the {web_fetch_tool_name} tool"
            )

        formatted_response = json.dumps(search_response.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        log.info(f"searxng_web_search tool call completed with FitSearXNGResponse:\n{formatted_response}")

    except Exception:
        msg = "An error occurred while attempting to use SearXNG to search the web"
        log.error(msg)
        raise RuntimeError(f"{msg}:\n{traceback.format_exc()}")

    return search_response


def main() -> int:
    args = parse_args()

    setup_logger(args)

    log.info("Starting MCP SearXNG server")

    set_mcp_vars(args)
    validate_mcp_vars()

    mcp.run(show_banner=False)

    log.info("MCP SearXNG server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
