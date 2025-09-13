"""
AI Service Module - Multi-Model LangChain Implementation

This module provides AI services supporting multiple LLM providers:
- Ollama (local models)  
- OpenAI (GPT models)
- Anthropic (Claude models)
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# Load environment variables
load_dotenv()


class ModelProvider(Enum):
    """Supported model providers"""
    AUTO = "auto"      # Ollama (local)
    GPT = "gpt"        # OpenAI
    CLAUDE = "claude"  # Anthropic


class AIService:
    """AI Service for handling multiple LLM providers"""
    
    def __init__(self):
        self.chains = {}
        self._setup_config()
    
    def _setup_config(self):
        """Setup configuration from environment variables"""
        self.config = {
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'model': os.getenv('OLLAMA_DEFAULT_MODEL', 'qwen3:8b')
            },
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                'model': 'gpt-3.5-turbo'
            },
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'model': 'claude-3-haiku-20240307'
            }
        }
    
    def create_prompt_template(self, custom_template: Optional[str] = None) -> PromptTemplate:
        """
        Create a prompt template for the conversation.
        
        Args:
            custom_template: Optional custom template string
            
        Returns:
            PromptTemplate: Configured prompt template
        """
        template = custom_template or "Human: {question}\n\nAssistant:"
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )
    
    def create_ollama_chain(self, model: Optional[str] = None) -> RunnableSequence:
        """
        Create LangChain chain with Ollama.
        
        Args:
            model: Optional model name, defaults to config
            
        Returns:
            RunnableSequence: Configured chain
        """
        model_name = model or self.config['ollama']['model']
        
        llm = OllamaLLM(
            model=model_name,
            base_url=self.config['ollama']['base_url']
        )
        
        prompt = self.create_prompt_template()
        return RunnableSequence(prompt, llm)
    
    def create_openai_chain(self, model: Optional[str] = None) -> RunnableSequence:
        """
        Create LangChain chain with OpenAI.
        
        Args:
            model: Optional model name, defaults to config
            
        Returns:
            RunnableSequence: Configured chain
        """
        if not self.config['openai']['api_key']:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        model_name = model or self.config['openai']['model']
        
        llm = ChatOpenAI(
            model=model_name,
            api_key=self.config['openai']['api_key'],
            base_url=self.config['openai']['base_url']
        )
        
        prompt = self.create_prompt_template()
        return RunnableSequence(prompt, llm)
    
    def create_anthropic_chain(self, model: Optional[str] = None) -> RunnableSequence:
        """
        Create LangChain chain with Anthropic.
        
        Args:
            model: Optional model name, defaults to config
            
        Returns:
            RunnableSequence: Configured chain
        """
        if not self.config['anthropic']['api_key']:
            raise ValueError("Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable.")
        
        model_name = model or self.config['anthropic']['model']
        
        llm = ChatAnthropic(
            model=model_name,
            api_key=self.config['anthropic']['api_key']
        )
        
        prompt = self.create_prompt_template()
        return RunnableSequence(prompt, llm)
    
    def get_chain(self, provider: ModelProvider, model: Optional[str] = None) -> RunnableSequence:
        """
        Get or create a chain for the specified provider.
        
        Args:
            provider: Model provider enum
            model: Optional specific model name
            
        Returns:
            RunnableSequence: Configured chain
        """
        cache_key = f"{provider.value}_{model or 'default'}"
        
        if cache_key not in self.chains:
            if provider == ModelProvider.AUTO:
                self.chains[cache_key] = self.create_ollama_chain(model)
            elif provider == ModelProvider.GPT:
                self.chains[cache_key] = self.create_openai_chain(model)
            elif provider == ModelProvider.CLAUDE:
                self.chains[cache_key] = self.create_anthropic_chain(model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        return self.chains[cache_key]
    
    def invoke_model(self, prompt: str, provider: ModelProvider, model: Optional[str] = None) -> str:
        """
        Invoke a model with the given prompt.
        
        Args:
            prompt: User's input prompt
            provider: Model provider to use
            model: Optional specific model name
            
        Returns:
            str: AI response
        """
        try:
            chain = self.get_chain(provider, model)
            response = chain.invoke({"question": prompt})
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"Error with {provider.value}: {str(e)}"
            
            # Add specific troubleshooting based on provider
            if provider == ModelProvider.AUTO:
                error_msg += "\n\nTroubleshooting:\n- Make sure Ollama is running (ollama serve)\n- Check if model is available (ollama list)"
            elif provider == ModelProvider.GPT:
                error_msg += "\n\nTroubleshooting:\n- Check OpenAI API key\n- Verify internet connection\n- Check API quotas"
            elif provider == ModelProvider.CLAUDE:
                error_msg += "\n\nTroubleshooting:\n- Check Anthropic API key\n- Verify internet connection\n- Check API quotas"
            
            raise Exception(error_msg)


# Global AI service instance
ai_service = AIService()


def get_ai_response(prompt: str, model_provider: str, specific_model: Optional[str] = None) -> str:
    """
    Convenience function to get AI response.
    
    Args:
        prompt: User's input prompt
        model_provider: Provider name ('auto', 'gpt', 'claude')
        specific_model: Optional specific model name
        
    Returns:
        str: AI response
    """
    provider = ModelProvider(model_provider.lower())
    return ai_service.invoke_model(prompt, provider, specific_model)


def main():
    """Interactive CLI for testing the AI service"""
    print("🤖 Multi-Model AI Service")
    print("Supports: Auto (Ollama), GPT (OpenAI), Claude (Anthropic)")
    print("Type 'quit' to exit, 'switch' to change model\n")
    
    current_provider = ModelProvider.AUTO
    
    while True:
        print(f"Current model: {current_provider.value}")
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'switch':
            print("Available providers: auto, gpt, claude")
            new_provider = input("Select provider: ").strip().lower()
            try:
                current_provider = ModelProvider(new_provider)
                print(f"Switched to {current_provider.value}")
            except ValueError:
                print("Invalid provider. Using current provider.")
            continue
        
        if not user_input:
            print("Please enter a question.")
            continue
        
        try:
            print("🤔 Thinking...")
            response = ai_service.invoke_model(user_input, current_provider)
            print(f"🤖 Assistant ({current_provider.value}): {response}\n")
        except Exception as e:
            print(f"❌ {e}\n")


if __name__ == "__main__":
    main()
