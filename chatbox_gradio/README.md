# 🦙 LangChain + Ollama + Gradio Chat

A beautiful web-based chat interface for interacting with Ollama using LangChain, built with Gradio.

## ✨ Features

- **Beautiful Web UI**: Modern, responsive chat interface
- **Real-time Chat**: Interactive conversation with Ollama models/OpenAI - GPT/Anthropic - Claude
- **LangChain Integration**: Structured prompts and chain management
- **Example Prompts**: Quick-start examples to get you chatting
- **Error Handling**: Graceful error handling with helpful messages
- **Auto-launch**: Opens in your browser automatically

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull the model**: 
   ```bash
   ollama pull qwen3:8b
   ```
3. **Start Ollama service**:
   ```bash
   ollama serve
   ```

### Running the Application

1. **Install dependencies** (if not already done):
   ```bash
   uv sync
   ```

2. Create `.env` file (refer to env.example)

3. **Start API backend**:
   ```bash
   uv run python backend.py
   ```

4. **Launch the Gradio interface**:
   ```bash
   # Simple interface (default)
   uv run python frontend.py
   ```

3. **Chat!** The interface will open in your browser at `http://localhost:7860`

