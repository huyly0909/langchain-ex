import gradio as gr
import requests
import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = f"http://{os.getenv('FLASK_HOST', '127.0.0.1')}:{os.getenv('FLASK_PORT', 5000)}"

def send_chat_request(prompt: str, model: str) -> str:
    """Send chat request to Flask backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"prompt": prompt, "model": model},
            timeout=60*3
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"Connection error: {str(e)}"

def chat_fn(message: str, history: List[List[str]], model: str) -> Tuple[List[List[str]], str]:
    """Process chat message"""
    if not message.strip():
        return history, ""
    
    # Extract model ID from selection
    model_id = "auto" if "Ollama" in model else "gpt" if "GPT" in model else "claude"
    
    history.append([message, "🤔 Thinking..."])
    response = send_chat_request(message, model_id)
    history[-1][1] = response
    
    return history, ""

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft(), title="🤖 Multi-Model AI Chat") as interface:
        gr.Markdown("# 🤖 Multi-Model AI Chat")
        gr.Markdown("*Chat with Ollama, OpenAI GPT, and Anthropic Claude*")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, avatar_images=("../data/User.png", "../data/AI Assistant.png"))
                with gr.Row():
                    with gr.Column(scale=3):
                        msg_input = gr.Textbox(placeholder="Type your message...", scale=4)
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear Chat 🗑️")
                
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["✅ Auto (Ollama)", "❓ GPT (OpenAI)", "❓ Claude (Anthropic)"],
                    value="✅ Auto (Ollama)",
                    label="AI Model"
                )
                
                gr.Markdown("### 💡 Examples")
                examples = [
                    "Hello! How are you?",
                    "Explain quantum computing",
                    "Write a Python function",
                    "Tell me a joke"
                ]
                
                for example in examples:
                    example_btn = gr.Button(example, size="sm")
                    example_btn.click(lambda x=example: x, outputs=[msg_input])
        
        # Wire up interface
        send_btn.click(chat_fn, [msg_input, chatbot, model_dropdown], [chatbot, msg_input])
        msg_input.submit(chat_fn, [msg_input, chatbot, model_dropdown], [chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])
    
    return interface

def main():
    """Launch frontend"""
    print(f"🚀 Starting Multi-Model Chat Frontend...")
    print(f"🔌 Backend: {BACKEND_URL}")
    
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv('GRADIO_PORT', 7860)),
        inbrowser=True
    )

if __name__ == "__main__":
    main()
