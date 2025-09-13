"""
Flask Backend API for Multi-Model AI Chat

This module provides a REST API for interacting with multiple LLM providers
through the AI service. Supports Ollama, OpenAI, and Anthropic models.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import Dict, Any
import logging

import sys
import os
sys.path.append(os.path.dirname(__file__))
from ai import get_ai_response, ModelProvider

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Chat Backend',
        'version': '1.0.0'
    })


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available model providers"""
    return jsonify({
        'models': [
            {
                'id': 'auto',
                'name': 'Auto (Ollama)',
                'provider': 'ollama',
                'description': 'Local Ollama models (qwen3:8b)',
                'requires_api_key': False
            },
            {
                'id': 'gpt',
                'name': 'GPT (OpenAI)',
                'provider': 'openai',
                'description': 'OpenAI GPT models',
                'requires_api_key': True,
                'available': bool(os.getenv('OPENAI_API_KEY'))
            },
            {
                'id': 'claude',
                'name': 'Claude (Anthropic)',
                'provider': 'anthropic',
                'description': 'Anthropic Claude models',
                'requires_api_key': True,
                'available': bool(os.getenv('ANTHROPIC_API_KEY'))
            }
        ]
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint that processes user prompts and returns AI responses.
    
    Expected JSON payload:
    {
        "prompt": "User's question or message",
        "model": "auto" | "gpt" | "claude",
        "specific_model": "optional specific model name"
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'prompt' not in data or not data['prompt'].strip():
            return jsonify({
                'error': 'Prompt is required and cannot be empty',
                'status': 'error'
            }), 400
        
        if 'model' not in data:
            return jsonify({
                'error': 'Model is required',
                'status': 'error'
            }), 400
        
        prompt = data['prompt'].strip()
        model_provider = data['model'].lower()
        specific_model = data.get('specific_model')
        
        # Validate model provider
        valid_providers = ['auto', 'gpt', 'claude']
        if model_provider not in valid_providers:
            return jsonify({
                'error': f'Invalid model provider. Must be one of: {valid_providers}',
                'status': 'error'
            }), 400
        
        logger.info(f"Processing chat request: model={model_provider}, prompt_length={len(prompt)}")
        
        # Get AI response
        response = get_ai_response(
            prompt=prompt,
            model_provider=model_provider,
            specific_model=specific_model
        )
        
        return jsonify({
            'response': response,
            'model_used': model_provider,
            'specific_model': specific_model,
            'status': 'success'
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        return jsonify({
            'error': f'Failed to process chat request: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint (placeholder for future implementation)
    """
    return jsonify({
        'error': 'Streaming not yet implemented',
        'status': 'error'
    }), 501


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


def create_app(config=None):
    """Application factory for testing"""
    if config:
        app.config.update(config)
    return app


def main():
    """Run the Flask development server"""
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Flask AI Backend Server...")
    print(f"🔗 Server: http://{host}:{port}")
    print(f"📋 Debug mode: {debug}")
    print(f"🤖 Supported models: Auto (Ollama), GPT (OpenAI), Claude (Anthropic)")
    
    # Check API key availability
    openai_available = bool(os.getenv('OPENAI_API_KEY'))
    anthropic_available = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    print(f"🔑 OpenAI API: {'✅ Available' if openai_available else '❌ Not configured'}")
    print(f"🔑 Anthropic API: {'✅ Available' if anthropic_available else '❌ Not configured'}")
    print(f"🦙 Ollama: Always available (local)\n")
    
    print("API Endpoints:")
    print("- GET  /health - Health check")
    print("- GET  /api/models - Available models")
    print("- POST /api/chat - Chat with AI")
    print("")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")


if __name__ == '__main__':
    main()
