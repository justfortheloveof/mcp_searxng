import argparse
import os
import traceback
from dataclasses import dataclass
from typing import Annotated, Any, cast
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP

searxng_url = os.getenv("SEARXNG_URL", "")

mcp = FastMCP("mcp_searxng")


@dataclass
class CLI_Args_mcp_searxng:
    server_url: str | None = None
    override_env: bool = False


def parse_args() -> CLI_Args_mcp_searxng:
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

    args = parser.parse_args()
    server_url = cast(str | None, args.server_url)
    override_env = cast(bool, args.override_env)

    if override_env and not server_url:
        parser.error("--override-env requires --server-url to be set")

    return CLI_Args_mcp_searxng(server_url=server_url, override_env=override_env)


def set_mcp_vars(args: CLI_Args_mcp_searxng) -> None:
    """Set SearXNG server URL based on CLI arguments."""
    # TODO: Support auth
    global searxng_url  # already set to either the value of env. var. "SEARXNG_URL" or "" at module level
    if (not searxng_url or args.override_env) and args.server_url:
        searxng_url = args.server_url


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


# TODO: don't use Any
def cleanup_search_results(search_results: dict[str, Any]) -> None:
    """Recursively delete a nested key from a dictionary using the dot notation."""
    # keys to remove from answer - control context content and size
    remove_keys = [
        "answers",
        "corrections",
        "infoboxes",
        "number_of_results",
        "suggestions",
        "unresponsive_engines",
        "results.category",
        "results.engines",
        "results.iframe_src",
        "results.img_src",
        "results.parsed_url",
        "results.positions",
        "results.priority",
        "results.publishedDate",
        "results.template",
        "results.thumbnail",
    ]

    # FIXME: crappy logic
    for key in remove_keys:
        if "." not in key:
            # top level
            if key in search_results:
                del search_results[key]
                continue

        # nested in results list
        keys = key.split(".")
        if keys[0] in search_results and isinstance(current := search_results[keys[0]], list):
            for item in current:
                if isinstance(item, dict) and keys[1] in item:
                    del item[keys[1]]


@mcp.tool
async def searxng_web_search(query: Annotated[str, "The web search query string"]):
    """Search the web"""
    # NOTE: hard coded params - notice we only ever return the first page of results
    # https://docs.searxng.org/dev/search_api.html
    search_params = {
        "q": query,
        "language": "en",
        "pageno": 1,
        "safesearch": 0,
        "format": "json",
        "engines": ["duckduckgo"],  # TODO: should be CLI arg - duckduckgo > brave are the only 2 tested
    }

    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(f"{searxng_url}/search", params=search_params, timeout=10.0)
        _ = response.raise_for_status()
        searxng_search_results = response.json()

        if not isinstance(searxng_search_results, dict):
            raise ValueError(f"Unexpected SearXNG ({searxng_url}) response format: {searxng_search_results}")

        cleanup_search_results(searxng_search_results)
        searxng_search_results["hint"] = (  # TODO: the name of the tool should be a CLI arg
            "These are the web search results for your query. Each result is a web page and "
            "you can access its whole content using the url value with the webfetch tool"
        )

        # import json
        # print(json.dumps(searxng_search_results, ensure_ascii=False, indent=4))
        # with open("searxng_response.json", "w", encoding="utf-8") as file:
        #     json.dump(searxng_search_results, file, ensure_ascii=False, indent=4)

    except Exception:
        raise RuntimeError(
            f"An error occurred while attempting to use SearXNG to search the web:\n{traceback.format_exc()}"
        )

    return searxng_search_results


def main() -> int:
    args = parse_args()

    set_mcp_vars(args)
    validate_mcp_vars()

    mcp.run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
