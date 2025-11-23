import asyncio
import json
import logging
import ssl
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from httpx import BasicAuth
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    engines: str = Field(
        default="duckduckgo,brave,startpage",
        description="Comma-separated list of SearXNG engines to use, as per SearXNG configuration",
    )
    engines_rotate: bool = Field(
        default=False,
        description=(
            "Whether to rotate through engines one at a time in round-robin fashion "
            "instead of querying all simultaneously"
        ),
    )
    include_engine: bool = Field(
        default=True,
        description="Whether to include the engine field in search results",
    )
    include_score: bool = Field(
        default=True,
        description="Whether to include the score field in search results",
    )
    include_hint: bool = Field(
        default=True,
        description="Whether to include a hint for the LLM in the search response",
    )
    hint: str = Field(
        default=(
            "Web search results for your query. Use the 'url' field with the webfetch tool to access page content. "
            "For more diverse sources, rerun the searxng_web_search tool with refined queries."
        ),
        description="The message you want sent to the llm with the search results as a hint of what to do with them",
    )
    ssl_verify: bool = Field(
        default=True,
        description="Whether to verify SSL certificates - unsafe",
    )
    # TODO: support custom SSL cert/CA directory (capath)
    ssl_ca_file: str | None = Field(
        default=None,
        description="Path to CA certificate file to trust for SSL verification",
    )
    auth_type: Literal["basic", "bearer", "api_key"] | None = Field(
        default=None,
        description="Authentication type for SearXNG server",
    )
    auth_username: str | None = Field(
        default=None,
        description="Username for basic authentication (required with --auth-type=basic)",
    )
    auth_password: str | None = Field(
        default=None,
        description="Password for basic authentication (required with --auth-type=basic)",
    )
    auth_token: str | None = Field(
        default=None,
        description="Bearer token for authentication (required with --auth-type=bearer)",
    )
    auth_api_key: str | None = Field(
        default=None,
        description="API key for authentication (required with --auth-type=api_key)",
    )
    auth_api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication",
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
                raise SystemExit(f"The directory for the log file '{path}' does not exist")
            if path.exists() and (not path.is_file() and not path.is_fifo()):
                raise SystemExit(f"The log file path must be a file, a symlink to a file or a fifo: {path}")
        return log_to

    @field_validator("engines")
    def validate_engines(cls, engines: str) -> str:
        engines = engines.strip()
        if " " in engines:
            raise SystemExit("--engines must not contain spaces")
        return engines

    @field_validator("ssl_ca_file")
    def validate_ssl_ca_file_path(cls, ssl_ca_file_path: str | None) -> str | None:
        if ssl_ca_file_path:
            path = Path(ssl_ca_file_path).resolve()
            if not path.exists():
                raise SystemExit(f"The SSL CA path does not exist: {path}")
            if not path.is_file():
                raise SystemExit(f"The SSL CA path must be a file or a symlink to a file: {path}")
        return ssl_ca_file_path

    @model_validator(mode="after")
    def validate_args(self):
        if self.override_env and not self.server_url:
            raise SystemExit("--override-env requires --server-url=URL")
        if self.log_level and not self.log_to:
            raise SystemExit("--log-to is required when --log-level is provided")

        if self.engines_rotate and len(self.engines.split(",")) < 2:
            raise SystemExit("--engines-rotate requires at least two engines to be provided with --engines")

        return self


class MCPSearXNGConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # cli args
    args: MCPSearXNGArgs
    # env vars
    env: MCPSearXNGEnvVars

    searxng_url: str = ""

    engine_rotation_index: int = 0
    rotation_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)

    @model_validator(mode="after")
    def set_and_validate_searxng_url(self):
        if (not self.env.URL or self.args.override_env) and self.args.server_url:
            self.searxng_url = self.args.server_url
        else:
            self.searxng_url = self.env.URL

        if not self.searxng_url:
            msg = (
                "SearXNG server URL is not set. "
                "Provide a valid URL via the SEARXNG_URL environment variable or command line argument"
            )
            log.critical(msg)
            raise SystemExit(msg)

        parsed_url = urlparse(self.searxng_url)

        if not all([parsed_url.scheme, parsed_url.netloc, parsed_url.scheme.lower() in ["http", "https"]]):
            msg = (
                f"Invalid SearXNG URL '{self.searxng_url}'. "
                "Please provide a valid URL (must start with http:// or https://)"
            )
            log.critical(msg)
            raise SystemExit(msg)

        if self.args.auth_type and parsed_url.scheme != "https":
            msg = "Authentication requires HTTPS for security. Please use an HTTPS URL"
            log.critical(msg)
            raise SystemExit(msg)

        if not self.args.ssl_verify and (self.args.ssl_ca_file or self.args.auth_type):
            raise SystemExit("--no-ssl-verify cannot be used with auth or when an SSL CA file is provided")

        # this could be in MCPSearXNGArgs, but its cleaner to validate these after checking for HTTPS
        if self.args.auth_type == "basic":
            if not self.args.auth_username or not self.args.auth_password:
                raise SystemExit("--auth-type=basic requires --auth-username and --auth-password")
        elif self.args.auth_type == "bearer":
            if not self.args.auth_token:
                raise SystemExit("--auth-type=bearer requires --auth-token")
        elif self.args.auth_type == "api_key":
            if not self.args.auth_api_key:
                raise SystemExit("--auth-type=api_key requires --auth-api-key")

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


def redact_config_secrets(config: MCPSearXNGConfig) -> MCPSearXNGConfig:
    redacted_config = MCPSearXNGConfig(args=config.args.model_copy(), env=config.env)
    if redacted_config.args.auth_password:
        redacted_config.args.auth_password = "***password***"
    if redacted_config.args.auth_token:
        redacted_config.args.auth_token = "***token***"
    if redacted_config.args.auth_api_key:
        redacted_config.args.auth_api_key = "***apikey***"
    return redacted_config


async def _search_raw(search_params: SearXNGSearchParams) -> SearXNGResponse:
    url = f"{config.searxng_url}/search"

    log.info(f"requesting SearXNG search at {url} with params: {search_params}, timeout: {config.args.server_timeout}")

    verify = config.args.ssl_verify
    if config.args.ssl_ca_file:
        ctx = ssl.create_default_context(cafile=config.args.ssl_ca_file)
        verify = ctx

    auth = None
    headers: dict[str, str] = {}
    if config.args.auth_type == "basic":
        assert config.args.auth_username is not None
        assert config.args.auth_password is not None
        auth = BasicAuth(username=config.args.auth_username, password=config.args.auth_password)
    elif config.args.auth_type == "bearer":
        headers["Authorization"] = f"Bearer {config.args.auth_token}"
    elif config.args.auth_type == "api_key" and config.args.auth_api_key:
        headers[config.args.auth_api_key_header] = config.args.auth_api_key

    try:
        async with httpx.AsyncClient(verify=verify, auth=auth, headers=headers) as client:
            response = await client.get(
                url, timeout=config.args.server_timeout, params=search_params.model_dump(exclude_none=True)
            )

        log.info(f"Response received from SearXNG with status code: {response.status_code}")
        _ = response.raise_for_status()
        search_response = SearXNGResponse.model_validate(response.json())

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            msg = "Authentication to SearXNG server failed: Invalid credentials provided"
        elif exc.response.status_code == 403:
            msg = "Access to SearXNG server forbidden: Authentication may be required or credentials lack permission"
        else:
            msg = f"SearXNG request failed with status {exc.response.status_code}: {exc}"
        log.error(msg)
        raise ToolError(msg) from exc

    except Exception as exc:
        msg = f"SearXNG request failed: {exc}"
        log.error(msg)
        raise ToolError(msg) from exc

    log.debug(
        f"Validated Raw SearXNGResponse: {json.dumps(search_response.model_dump(), ensure_ascii=False, indent=2)}"
    )

    return search_response


async def _search(query: str) -> FitSearXNGResponse:
    def _validate_search_response(search_response: SearXNGResponse) -> None:
        """Check if the search response is valid, meaning at least 1 search engine was responsive"""
        if search_response.unresponsive_engines:
            log.warning(f"Unresponsive SearXNG engine(s): {search_response.unresponsive_engines}")

            if not search_response.results:
                msg = (
                    "It seems like all requested SearXNG engines were unresponsive: "
                    f"{search_response.unresponsive_engines}"
                )
                log.error(msg)
                raise ToolError(msg)

    def _clean_search_response(search_response: FitSearXNGResponse) -> FitSearXNGResponse:
        """Adjust fields based on CLI args"""
        for result in search_response.results:
            if not config.args.include_engine:
                result.engine = None
            if not config.args.include_score:
                result.score = None

        return search_response

    if not config.args.engines_rotate:
        search_params = SearXNGSearchParams(q=query, engines=config.args.engines)
        search_response = await _search_raw(search_params)
        _validate_search_response(search_response)
        search_response = _clean_search_response(search_response)
        fit_search_response = FitSearXNGResponse.model_validate(search_response.model_dump(exclude_none=True))

        return fit_search_response

    else:
        engines_list = [e.strip() for e in config.args.engines.split(",") if e.strip()]
        if not engines_list:
            raise ToolError("No engines configured for rotation")

        # Cycle through engines
        for _ in range(len(engines_list)):
            async with config.rotation_lock:
                selected_engine = engines_list[config.engine_rotation_index % len(engines_list)]
                config.engine_rotation_index += 1

            log.debug(f"Trying engine: {selected_engine}")
            search_params = SearXNGSearchParams(q=query, engines=selected_engine)
            search_response = await _search_raw(search_params)

            unresponsive_names = [ue[0] for ue in (search_response.unresponsive_engines or [])]
            if search_response.results and selected_engine not in unresponsive_names:
                log.debug(f"Engine {selected_engine} succeeded")
                search_response = _clean_search_response(search_response)
                fit_search_response = FitSearXNGResponse.model_validate(search_response.model_dump(exclude_none=True))

                return fit_search_response
            else:
                log.debug(f"Engine {selected_engine} failed or unresponsive, trying next")

        log.warning("All engines failed, falling back to category-based engines")
        fallback_params = SearXNGSearchParams(q=query, engines="")
        fallback_response = await _search_raw(fallback_params)
        _validate_search_response(fallback_response)
        search_response = _clean_search_response(fallback_response)
        fit_fallback_response = FitSearXNGResponse.model_validate(fallback_response.model_dump(exclude_none=True))

        return fit_fallback_response


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

    search_response = await _search(query)

    if config.args.include_hint:
        search_response_dict = search_response.model_dump(exclude_none=True)
        search_response_dict["hint"] = config.args.hint
        search_response = FitSearXNGResponseWithHint.model_validate(search_response_dict)

    formatted_response = json.dumps(search_response.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
    log.info(f"searxng_web_search tool call completed with FitSearXNGResponse: {formatted_response}")

    return search_response


def main() -> int:
    global config
    config = MCPSearXNGConfig(args=MCPSearXNGArgs(), env=MCPSearXNGEnvVars())

    setup_logger(config)

    log.info("Starting MCP SearXNG server")
    redacted_config = redact_config_secrets(config).model_dump(exclude={"rotation_lock"})
    log.debug(f"MCP SearXNG started with config: {json.dumps(redacted_config, ensure_ascii=False, indent=2)}")

    mcp.run(show_banner=False)

    log.info("MCP SearXNG server stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
