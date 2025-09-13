# langchain-ex

A LangChain application that integrates with Ollama to provide an interactive chat experience.

## Features

- Interactive chatbot using LangChain and Ollama
- Uses qwen3:8b model (configurable)
- Simple prompt-response loop with graceful error handling

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai)
2. **Pull the model**: Run `ollama pull qwen3:8b` to download the model
3. **Start Ollama**: Run `ollama serve` to start the Ollama service

## Installation

This project uses `uv` for Python package management:

```bash
# Install dependencies
uv sync

# Or if you prefer pip
uv pip install -e .
```

## Usage

Run the interactive chat:

```bash
uv run python main.py
```

The application will:
1. Ask you to enter a prompt
2. Send your question to Ollama via LangChain
3. Print the AI response
4. Allow you to continue the conversation

Type `quit`, `exit`, or `q` to stop the conversation.
