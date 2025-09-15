## Setup Taiga MCP Server (SSE)
1. clone the `pytaiga-mcp` repository
```bash
git clone https://github.com/talhaorak/pytaiga-mcp
cd pytaiga-mcp
uv sync
```

2. launch Taiga MCP Server
```bash

uv run python src/server.py
```

## Setup AI
1. uv sync
2. create `.env` following env.sse.example
3. start the conversation
```bash
uv run python main.py
```
4. enjoy