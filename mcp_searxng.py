import json
import logging
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError


log = logging.getLogger(__name__)
mcp = FastMCP("mcp_searxng")


class MCPSearXNGEnvVars(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="SEARXNG_",
        case_sensitive=True,
    )

    # TODO: support all or most MCPSearXNGArgs fields as ENV VARS?
    URL: str = ""


class MCPSearXNGArgs(BaseSettings):
    # TODO: look into re-introducing arg_parse for potentially better UX
    # https://docs.pydantic.dev/latest/concepts/pydantic_settings/#integrating-with-existing-parsers
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        case_sensitive=True,
        cli_parse_args=True,  # replace typical arg_parse
        cli_implicit_flags=True,  # bool args automatically have a --no equivalent
        cli_avoid_json=True,  # display None on CLI instead of null
        cli_enforce_required=True,  # enforce CLI to provide fields with no default value
        cli_hide_none_type=True,
        cli_kebab_case=True,
        nested_model_default_partial_update=True,
    )

    # TODO: Support auth
    # TODO: support custom SSL cert/CA

    server_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("s", "server_url"),
        description="SearXNG server URL (env. SEARXNG_URL)",
    )
    server_timeout: int | float | None = Field(
        default=5,
        validation_alias=AliasChoices("t", "server_timeout"),
        description="SearXNG server timeout",
    )
    override_env: bool = Field(
        default=False,
        validation_alias=AliasChoices("o", "override_env"),
        description="Whether to override environment variables with the CLI arguments",
    )
    # TODO: add rotate engines switch, to use the engines 1 at a time
    engines: str = Field(
        default="duckduckgo,brave,startpage,google",
        description="Comma-separated list of SearXNG engines to use",
    )
    include_hint: bool = Field(
        default=True,
        description="Whether to include a hint in the search response about using the web fetch tool",
    )
    web_fetch_tool_name: str = Field(
        default="webfetch",
        description="Name of the web fetch tool to include in the hint",
    )
    ssl_verify: bool = Field(
        default=True,
        description="Whether to verify SSL certificates - unsafe",
    )
    log_to: str | None = Field(
        default=None,
        description="Path to log file - enables logging",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = Field(
        default=None,
        description="Logging level",
    )

    @field_validator("log_to")
    def validate_log_to(cls, log_to: str | None) -> str | None:
        if log_to:
            path = Path(log_to).resolve()
            if not path.parent.exists():
                raise SystemExit(f"The directory for the log file '{path}' does not exist.")
        return log_to

    @field_validator("engines")
    def validate_engines(cls, engines: str) -> str:
        engines = engines.strip()
        if " " in engines:
            raise SystemExit("--engines must not contain spaces")
        return engines

    @model_validator(mode="after")
    def validate_args(self):
        if self.override_env and not self.server_url:
            raise SystemExit("--override-env requires --server-url URL to be provided")
        if self.log_level and not self.log_to:
            raise SystemExit("--log-to is required when --log-level is provided")

        return self


class MCPSearXNGConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    # cli args
    args: MCPSearXNGArgs
    # env vars
    env: MCPSearXNGEnvVars
    # set later
    searxng_url: str = ""

    @model_validator(mode="after")
    def set_and_validate_searxng_url(self):
        if (not self.env.URL or self.args.override_env) and self.args.server_url:
            self.searxng_url = self.args.server_url
        else:
            self.searxng_url = self.env.URL

        if not self.searxng_url:
            msg = (
                "SearXNG server URL is not set. "
                "Please provide a valid URL via the SEARXNG_URL environment variable or command line argument."
            )
            log.critical(msg)
            raise SystemExit(msg)

        parsed_url = urlparse(self.searxng_url)

        if not all([parsed_url.scheme, parsed_url.netloc]):
            msg = f"Invalid SearXNG URL '{self.searxng_url}'. Please provide a valid URL."
            log.critical(msg)
            raise SystemExit(msg)

        return self


class FitSearXNGResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    url: str | None = Field(None, description="The URL of the search result")
    title: str | None = Field(None, description="The title of the search result")
    content: str | None = Field(None, description="The content snippet of the search result")
    engine: str | None = Field(None, description="The search engine that provided this result")
    score: float | None = Field(None, description="The relevance score of the result")


class SearXNGResult(FitSearXNGResult):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

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


class FitSearXNGResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    query: str = Field(description="The search query string")
    results: list[FitSearXNGResult] = Field(description="List of search results")


class FitSearXNGResponseWithHint(FitSearXNGResponse):
    hint: str = Field(description="A hint to influence AI / LLMs behavior")


class SearXNGResponse(FitSearXNGResponse):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    number_of_results: int | None = None
    answers: list[dict[str, Any]] | None = None  # pyright: ignore[reportExplicitAny]
    corrections: list[str] | None = None
    infoboxes: list[dict[str, Any]] | None = None  # pyright: ignore[reportExplicitAny]
    suggestions: list[str] | None = None
    unresponsive_engines: list[list[str]] | None = None


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
    engines: str  # config.args.engines


def setup_logger(config: MCPSearXNGConfig) -> None:
    if config.args.log_to:
        logging.basicConfig(
            level=logging.getLevelNamesMapping().get(config.args.log_level or "DEBUG"),
            format="[%(asctime)s %(process)d %(levelname)s] %(message)s",
            handlers=[logging.FileHandler(config.args.log_to)],
        )


async def _search(search_params: SearXNGSearchParams) -> FitSearXNGResponse:
    url = f"{config.searxng_url}/search"

    log.info(f"requesting SearXNG search at {url} with params: {search_params}, timeout: {config.args.server_timeout}")

    try:
        async with httpx.AsyncClient(verify=config.args.ssl_verify) as client:
            response = await client.get(
                url, timeout=config.args.server_timeout, params=search_params.model_dump(exclude_none=True)
            )

        log.info(f"Response received from SearXNG with status code: {response.status_code}")
        _ = response.raise_for_status()

    except Exception as exc:
        msg = f"SearXNG request failed: {exc}"
        log.error(msg)
        raise ToolError(msg) from exc

    search_response = SearXNGResponse.model_validate(response.json())

    _validate_search_response(search_params, search_response)
    log.debug(
        f"Validated Raw SearXNGResponse: {json.dumps(search_response.model_dump(), ensure_ascii=False, indent=2)}"
    )
    fit_search_response = FitSearXNGResponse.model_validate(search_response.model_dump())

    return fit_search_response


def _validate_search_response(search_params: SearXNGSearchParams, search_response: SearXNGResponse) -> None:
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
            raise ToolError(msg)


@mcp.tool
async def searxng_web_search(
    query: Annotated[
        str,
        (
            "The web search query string. Note that you can negate search terms with a leading minus ('-') symbol. "
            "For example 'python best practices -reddit.com' will search for 'python best practices' and remove "
            "results from 'reddit.com'."
        ),
    ],
) -> FitSearXNGResponse | FitSearXNGResponseWithHint:
    """Search the web"""
    log.info(f"searxng_web_search tool called with query: '{query}'")

    if query.strip() == "":
        log.error("Search query is an empty string")
        raise ToolError("The 'query' field cannot be empty")

    search_params = SearXNGSearchParams(q=query, engines=config.args.engines)

    search_response = await _search(search_params)

    if config.args.include_hint:
        search_response_dict = search_response.model_dump(exclude_none=True)
        search_response_dict["hint"] = (
            "These are the web search results for your query. Each result is a web page and you can access its whole "
            f"content using the url value with the {config.args.web_fetch_tool_name} tool. If the search results are "
            "less than satisfactory, consider running the searxng_web_search tool again with a different query."
        )
        search_response = FitSearXNGResponseWithHint.model_validate(search_response_dict)

    formatted_response = json.dumps(search_response.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
    log.info(f"searxng_web_search tool call completed with FitSearXNGResponse: {formatted_response}")

    return search_response


def main() -> int:
    global config
    config = MCPSearXNGConfig(args=MCPSearXNGArgs(), env=MCPSearXNGEnvVars())

    setup_logger(config)

    log.info("Starting MCP SearXNG server")
    log.debug(f"MCP SearXNG started with config: {json.dumps(config.model_dump(), ensure_ascii=False, indent=2)}")

    mcp.run(show_banner=False)

    log.info("MCP SearXNG server stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
