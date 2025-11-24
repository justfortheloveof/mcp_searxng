# MCP SearXNG

A low token consumption Python MCP server for interacting with a local or remote SearXNG instance.

## Features

📉 **Token Efficient** – Low token consumption, structured format, full user control  
💡 **Context Hints** – Customizable guidance for the LLM in server responses  
⚙️ **Engine Control** – Configure exactly which SearXNG engines are queried  
🔄 **Engine Rotation** – Optional round-robin engine cycling  
🎯 **Smart Filtering** - Automatically filters out unresponsive SearXNG engines  
🔐 **Authenticated** – Support for private/secured SearXNG instances (Basic, Bearer, API Key)  
🛡️ **Secure** – SSL verification with optional custom certificate support  
📝 **Logs** – Comprehensive and configurable logging to review tool usage flow

## Installation

The recommended way to install and/or use this tool is with `uv`

### From GitHub (Users)

This installs `mcp-searxng` globally

```sh
uv tool install git+https://github.com/justfortheloveof/mcp_searxng
```

#### Upgrade to Latest Version

```sh
uv tool upgrade mcp-searxng
```

### From Source (Development)

This installs `mcp-searxng` globally

```sh
# clone the repository
git clone https://github.com/justfortheloveof/mcp_searxng.git
cd mcp_searxng

# install as a tool
uv tool install .
```

## No Installation

This runs `mcp-searxng` without installing it

```sh
uvx --from git+https://github.com/justfortheloveof/mcp_searxng mcp-searxng --server-url "https://example.server"
```

## Usage

### Basic Usage

```sh
# Point to your SearXNG instance
mcp-searxng --server-url "https://server.example"
```

### Configuration

You can configure the server using command-line arguments or environment variables.

```help
usage: mcp_searxng.py [-h] [-s str] [-t {int,float}] [-o | --override-env | --no-override-env] [--engines str] [--engines-rotate | --no-engines-rotate] [--include-engine | --no-include-engine]
                      [--include-score | --no-include-score] [--include-hint | --no-include-hint] [--hint str] [--ssl-verify | --no-ssl-verify] [--ssl-ca-file str] [--auth-type {basic,bearer,api_key}]
                      [--auth-username str] [--auth-password str] [--auth-token str] [--auth-api-key str] [--auth-api-key-header str] [--log-to str] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

options:
  -h, --help            show this help message and exit
  -s, --server-url str  SearXNG server URL (env. SEARXNG_URL) (default: None)
  -t, --server-timeout {int,float}
                        SearXNG server timeout (default: 5)
  -o, --override-env, --no-override-env
                        Whether to override environment variables with the CLI arguments (default: False)
  --engines str         Comma-separated list of SearXNG engines to use, as per SearXNG configuration (default: duckduckgo,brave,startpage)
  --engines-rotate, --no-engines-rotate
                        Whether to rotate through engines one at a time in round-robin fashion instead of querying all simultaneously (default: False)
  --include-engine, --no-include-engine
                        Whether to include the engine field in search results (default: True)
  --include-score, --no-include-score
                        Whether to include the score field in search results (default: True)
  --include-hint, --no-include-hint
                        Whether to include a hint for the LLM in the search response (default: True)
  --hint str            The message you want sent to the llm with the search results as a hint of what to do with them (default: Web search results for your query. Use the 'url' field with the webfetch tool
                        to access page content. For more diverse sources, rerun the searxng_web_search tool with refined queries.)
  --ssl-verify, --no-ssl-verify
                        Whether to verify SSL certificates - unsafe (default: True)
  --ssl-ca-file str     Path to CA certificate file to trust for SSL verification (default: None)
  --auth-type {basic,bearer,api_key}
                        Authentication type for SearXNG server (default: None)
  --auth-username str   Username for basic authentication (required with --auth-type=basic) (default: None)
  --auth-password str   Password for basic authentication (required with --auth-type=basic) (default: None)
  --auth-token str      Bearer token for authentication (required with --auth-type=bearer) (default: None)
  --auth-api-key str    API key for authentication (required with --auth-type=api_key) (default: None)
  --auth-api-key-header str
                        Header name for API key authentication (default: X-API-Key)
  --log-to str          Path to log file - enables logging (default: None)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: None)
```

### Example with Opencode

With `mcp-searxng` installed:

```jsonc
{
  "mcp": {
    "searxng": {
      "enabled": true,
      "type": "local",
      "command": ["mcp-searxng", "--server-url", "https://example.server"]
    }
  }
}
```

With `uvx`

```jsonc
{
  "mcp": {
    "searxng": {
      "enabled": true,
      "type": "local",
      "command": [
        "uvx",
        "--from",
        "git+https://github.com/justfortheloveof/mcp_searxng mcp-searxng",
        "mcp-searxng",
        "--server-url",
        "https://example.server"
      ]
    }
  }
}
```
