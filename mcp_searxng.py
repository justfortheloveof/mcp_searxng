import argparse
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict, cast
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP


# TODO: add CLI Args to control logging level and logging to filepath
log = logging.getLogger(__name__)

searxng_url = os.getenv("SEARXNG_URL", "")
mcp = FastMCP("mcp_searxng")


class SearXNGResult(TypedDict, total=False):
    url: str | None
    title: str | None
    content: str | None
    engine: str | None
    score: float | None
    category: str | None
    parsed_url: dict | None  # Serialized ParseResult
    template: str | None
    positions: list[int] | None
    priority: Literal["", "high", "low"] | None
    thumbnail: str | None
    publishedDate: str | None  # ISO datetime string
    pretty_url: str | None
    img_src: str | None
    iframe_src: str | None
    audio_src: str | None
    pubdate: str | None
    length: str | None  # Serialized timedelta
    views: str | None
    author: str | None
    metadata: str | None
    engines: list[str] | None  # Serialized set
    open_group: bool | None
    close_group: bool | None


class SearXNGResponse(TypedDict, total=False):
    query: str
    results: list[SearXNGResult]
    number_of_results: int | None
    answers: list[dict] | None
    corrections: list[str] | None
    infoboxes: list[dict] | None
    suggestions: list[str] | None
    unresponsive_engines: list[str] | None
    hint: str | None  # Added by MCP server (us)


@dataclass
class CLI_Args:
    server_url: str | None = None
    override_env: bool = False
    include_hint: bool = False
    web_fetch_tool_name: str = "webfetch"


def parse_args() -> CLI_Args:
    parser = argparse.ArgumentParser(description="MCP server to search and read web URLs using SearXNG")

    _ = parser.add_argument(
        "--server-url",
        "-s",
        type=str,
        help=(
            "SearXNG server URL. "
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
        default=False,
        help="Include a hint in search results suggesting to use the web fetch tool.",
    )
    _ = parser.add_argument(
        "--web-fetch-tool-name",
        type=str,
        default="webfetch",
        help="Name of the web fetch tool to reference in LLM hint (default: webfetch).",
    )

    args = parser.parse_args()
    server_url = cast(str | None, args.server_url)
    override_env = cast(bool, args.override_env)
    include_hint = cast(bool, args.include_hint)
    web_fetch_tool_name = cast(str, args.web_fetch_tool_name)

    if override_env and not server_url:
        parser.error("--override-env requires --server-url to be set")

    return CLI_Args(
        server_url=server_url,
        override_env=override_env,
        include_hint=include_hint,
        web_fetch_tool_name=web_fetch_tool_name,
    )


def set_mcp_vars(args: CLI_Args) -> None:
    """Set SearXNG server URL based on CLI arguments."""
    # TODO: Support auth
    global searxng_url  # NOTE: already set to either the value of env. var. "SEARXNG_URL" or "" at module level
    if (not searxng_url or args.override_env) and args.server_url:
        searxng_url = args.server_url
        log.info(f"SearXNG server URL set from command line argument: {searxng_url}")
    else:
        log.info(f"SearXNG server URL set from environment variable: {searxng_url}")

    global include_hint
    global web_fetch_tool_name
    include_hint = args.include_hint
    web_fetch_tool_name = args.web_fetch_tool_name
    log.info(f"Include hint: {include_hint}, Web fetch tool name: {web_fetch_tool_name}")


def validate_mcp_vars() -> None:
    """Check that the SearXNG server URL is set and valid, raise otherwise"""
    if not searxng_url:
        raise ValueError(
            (
                "SearXNG server URL is not set. "
                "Please provide a valid URL via the SEARXNG_URL environment variable or command line argument."
            )
        )

    parsed_url = urlparse(searxng_url)

    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise ValueError(f"Invalid SearXNG URL '{searxng_url}'. Please provide a valid URL.")


def remove_keys_recursive(obj: object, keys: list[str]):
    """
    Recursively remove a key from a nested structure using a list of keys.
    Handles dictionaries and lists. '[]' in a key indicates list traversal (apply to each item).
    Modifies in-place.
    """
    if not keys:
        return
    key = keys[0]
    if key.endswith("[]"):
        # handle list traversal: strip '[]', ensure it's a list, and recurse into each item
        key = key[:-2]
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
            for item in obj[key]:  # pyright: ignore[reportUnknownVariableType]
                remove_keys_recursive(cast(object, item), keys[1:])
    else:
        # normal dict access
        if isinstance(obj, dict) and key in obj:
            if len(keys) == 1:
                del obj[key]
            else:
                remove_keys_recursive(cast(object, obj[key]), keys[1:])
        elif isinstance(obj, list):
            for item in obj:  # pyright: ignore[reportUnknownVariableType]
                remove_keys_recursive(cast(object, item), keys)


def cleanup_search_response(search_response: SearXNGResponse) -> None:
    """
    Remove specified keys from the search response to clean up the data and control LLM context content and size.
    Modifies search_response in-place.
    """
    remove_keys = [
        "answers",
        "corrections",
        "infoboxes",
        "number_of_results",
        "suggestions",
        "unresponsive_engines",
        "results[].category",
        "results[].engines",
        "results[].iframe_src",
        "results[].img_src",
        "results[].parsed_url",
        "results[].positions",
        "results[].priority",
        "results[].publishedDate",
        "results[].template",
        "results[].thumbnail",
    ]

    for key in remove_keys:
        keys = key.split(".")
        remove_keys_recursive(search_response, keys)


async def search(search_params: dict[str, str | int]) -> SearXNGResponse:
    url = f"{searxng_url}/search"
    timeout = 10.0

    log.info(f"requesting SearXNG search at {url} with params: {search_params}, timeout: {timeout}")

    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(url, timeout=timeout, params=search_params)

    _ = response.raise_for_status()
    log.info(f"Response received from SearXNG with status code {response.status_code}")

    # import json
    #
    # with open("searxng_response.json", "w", encoding="utf-8") as file:
    #     json.dump(response.json(), file, ensure_ascii=False, indent=4)

    search_response = cast(SearXNGResponse, response.json())
    log.debug(f"SearXNG response JSON: {search_response}")

    return search_response


def validate_search_response(search_params: dict[str, str | int], search_response: SearXNGResponse) -> None:
    """Check if the search response is valid, meaning at least 1 search engine was responsive"""
    if "unresponsive_engines" in search_response and search_response["unresponsive_engines"]:
        # we have a non empty list of unresponsive_engines
        log.warning(f"Unresponsive SearXNG engine(s): {search_response['unresponsive_engines']}")

        if len(search_response["unresponsive_engines"]) == len(cast(str, search_params["engines"]).split(",")):
            # all requested engines were unresponsive
            msg = f"All requested SearXNG engines were unresponsive: {search_response['unresponsive_engines']}"
            log.error(msg)
            raise RuntimeError(msg)


@mcp.tool
async def searxng_web_search(query: Annotated[str, "The web search query string"]) -> SearXNGResponse:
    """Search the web"""
    log.info(f"searxng_web_search called with query: {query}")

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
        # TODO: should be CLI arg - duckduckgo > brave are the only 2 tested
        "engines": "duckduckgo,brave,startpage",
    }

    try:
        search_response = await search(search_params)
        validate_search_response(search_params, search_response)

        # TODO: use pydantic?
        if not isinstance(search_response, dict):
            raise ValueError(f"Unexpected SearXNG ({searxng_url}) response format: {search_response}")

        cleanup_search_response(search_response)

        if include_hint:
            search_response["hint"] = (
                "These are the web search results for your query. Each result is a web page and "
                f"you can access its whole content using the url value with the {web_fetch_tool_name} tool"
            )

        log.info((f"SearXNG search completed with response:\n{search_response}"))

        # import json
        # print(json.dumps(searxng_search_results, ensure_ascii=False, indent=4))
        # with open("searxng_response.json", "w", encoding="utf-8") as file:
        #     json.dump(searxng_search_results, file, ensure_ascii=False, indent=4)

    except Exception:
        msg = "An error occurred while attempting to use SearXNG to search the web"
        log.error(msg)
        raise RuntimeError(f"{msg}:\n{traceback.format_exc()}")

    return search_response


def main() -> int:
    args = parse_args()

    set_mcp_vars(args)
    validate_mcp_vars()

    mcp.run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
