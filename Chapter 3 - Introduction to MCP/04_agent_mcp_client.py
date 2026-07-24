import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


# Load environment variables
load_dotenv()

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")

print("OpenAI API Key Found")


async def main():
    # Get the absolute path of the MCP server
    server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "math_mcp_server.py"
    )

    # Create MCP client configuration
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": [server_path],
            }
        }
    )

    # Discover tools from the MCP server
    tools = await client.get_tools()

    print("\nDiscovered MCP Tools:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Initialize the LLM
    model = init_chat_model(
        "gpt-5.5",
        model_provider="openai"
    )

    # Create LangChain agent using MCP tools
    agent = create_agent(
        model=model,
        tools=tools
    )

    # Run the agent
    question = "What is 15 multiplied by 8 and then divided by 3?"

    print("\nUser Question:")
    print(question)

    print("\nRunning Agent...\n")

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    # Print final response
    print("\nFinal Answer:")

    final_message = response["messages"][-1]

    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())