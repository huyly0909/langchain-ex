import asyncio
import os
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import SSEConnection
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 1) LLM Configuration
LLM = ChatAnthropic(
    model=os.getenv('ANTHROPIC_MODEL', "claude-sonnet-4-20250514"),
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

# from langchain_ollama import ChatOllama
# LLM = ChatOllama(
#     model=os.getenv('OLLAMA_MODEL', "llama3.1:8b"),
#     base_url=os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")
# )

# Configuration
TAIGA_URL = os.getenv('TAIGA_URL', "http://localhost:9000")
TAIGA_USERNAME = os.getenv('TAIGA_USERNAME', "admin")
TAIGA_PASSWORD = os.getenv('TAIGA_PASSWORD','admin')
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', "http://localhost:8000/sse")

# 2) Define MCP server with SSE transport (Server-Sent Events)
# SSE maintains persistent HTTP connection - sessions stay in server memory!
MCP_CLIENT = MultiServerMCPClient({
    "taiga": SSEConnection(
        transport="sse",
        url=MCP_SERVER_URL,
        timeout=10.0,
        sse_read_timeout=60.0
    )
})

async def setup_tools():
    """Setup MCP tools asynchronously"""
    try:
        tools = await MCP_CLIENT.get_tools()
        print(f"✅ Successfully loaded {len(tools)} MCP tools:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        return tools
    except Exception as e:
        print(f"❌ Failed to connect to MCP server via SSE at {MCP_SERVER_URL}")
        print(f"   Error: {e}")
        print("\n💡 Make sure your MCP server supports SSE transport and is running")
        print(f"   Expected SSE endpoint: {MCP_SERVER_URL}")
        print("   The server should support Server-Sent Events over HTTP")
        raise

async def interactive_mode(executor: AgentExecutor):
    """Run interactive mode where user can ask multiple questions"""
    print("\n🎯 Taiga MCP Agent is ready! (SSE Transport)")
    print("📡 Connected to MCP server via Server-Sent Events!")
    print("💡 Try commands like:")
    print("   - 'show me all projects'")
    print("   - 'create a new project called \"My Project\"'")  
    print("   - 'list user stories in project 1'")
    print("   - 'create a user story \"As a user I want...\" in project 1'")
    print("   - Type 'quit' or 'exit' to stop\n")
    
    chat_history = []
    
    while True:
        user_input = None
        try:
            user_input = input("Enter your task: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print(f"\n🤖 Processing: {user_input}")
            result = await executor.ainvoke({"input": user_input, "chat_history": chat_history})
            
            print("\n✅ Result:")
            # Print just the output, not the full result dictionary
            output = result['output'] if isinstance(result, dict) and 'output' in result else str(result)
            print(output)
            print("-" * 100)
            
            # Add to chat history in proper LangChain message format
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=output)
            ])
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            # Call logout tool
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=f"Got error: {str(e)}")
            ])
            continue

# 4) Build an agent that can call those tools
PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a Taiga project management assistant. You can manage projects, user stories, tasks, issues, epics, and milestones.

IMPORTANT AUTHENTICATION:
- If you are not authenticated, you MUST first call the 'login' tool to get a session_id
- If you have a session_id, DON'T login again
- session_id example: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
- Use the session_id from login for all subsequent tool calls
- If you get authentication errors, try logging in again
- With SSE transport, the server maintains persistent HTTP connections and sessions in memory!

TAIGA CREDENTIALS:
- URL: {TAIGA_URL}
- Username: {TAIGA_USERNAME}
- Password: {TAIGA_PASSWORD}

WORKFLOW:
1. If user asks for data that requires authentication (like listing projects), login first
2. Use the session_id from login response for all other tool calls
3. For project management tasks, break them down into logical steps
4. Always provide clear feedback about what you're doing

Available tools include login, logout, session_status, and full CRUD operations for projects, user stories, tasks, issues, epics, and milestones."""),
    ("placeholder", f"{{chat_history}}"),
    ("human", f"{{input}}"),
    ("placeholder", f"{{agent_scratchpad}}")
])

async def main():
    print("🔧 Setting up Taiga MCP Agent with SSE Transport...")
    print(f"📡 Connecting to SSE endpoint: {MCP_SERVER_URL}")
    print("✅ SSE transport keeps persistent HTTP connection to server!")
    
    try:
        # Load MCP tools as LangChain tools
        tools = await setup_tools()
        agent = create_tool_calling_agent(LLM, tools, PROMPT)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Start interactive mode
        await interactive_mode(executor)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
