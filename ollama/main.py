from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def create_ollama_chain():
    """Create a LangChain chain with Ollama using qwen3:8b model"""
    # Initialize Ollama LLM with qwen3:8b model
    llm = OllamaLLM(model="qwen3:8b", base_url="http://localhost:11434")
    
    # Create a simple prompt template
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Human: {question}\n\nAssistant:"
    )
    
    # Create the chain
    chain = prompt | llm
    return chain


def main():
    """Main interactive loop for the chatbot"""
    print("🦙 Welcome to LangChain + Ollama Chat!")
    print("Using model: qwen3:8b")
    print("Type 'quit', 'exit', or 'q' to stop the conversation.\n")
    
    try:
        # Create the chain
        chain = create_ollama_chain()
        
        while True:
            # Step 1: Ask user to enter prompt
            user_input = input("You: ").strip()
            
            # Check for exit conditions
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or type 'quit' to exit.")
                continue
            
            try:
                # Step 2: Combine user inquiry and ollama to get response
                print("🤔 Thinking...")
                response = chain.invoke({"question": user_input})
                
                # Step 3: Print out the answer
                print(f"🦙 Assistant: {response}\n")
                
            except Exception as e:
                print(f"❌ Error getting response: {e}")
                print("Make sure Ollama is running and the qwen3:8b model is available.\n")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing the application: {e}")
        print("Make sure Ollama is running and the qwen3:8b model is available.")


if __name__ == "__main__":
    main()
