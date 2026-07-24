import asyncio
import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain_mcp_adapters.client import MultiServerMCPClient


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Check OpenAI API key
# --------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not configured. "
        "Add it to your .env file."
    )

print("OpenAI API Key Found")


# --------------------------------------------------
# 3. Connect to MCP Server
# --------------------------------------------------

async def create_mcp_client():

    # Get absolute path to MCP server
    server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "math_mcp_server.py"
    )

    # Configure MCP server
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": [server_path],
            }
        }
    )

    return client


# --------------------------------------------------
# 4. Run Agent
# --------------------------------------------------

async def main():

    print("\nConnecting to Math MCP Server...")

    client = await create_mcp_client()

    # --------------------------------------------------
    # Discover tools from MCP server
    # --------------------------------------------------

    tools = await client.get_tools()

    print("\nDiscovered MCP Tools:")

    for tool in tools:
        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )

    # --------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------

    model = init_chat_model(
        "gpt-5.5",
        model_provider="openai"
    )

    # --------------------------------------------------
    # Create LangChain Agent
    # --------------------------------------------------

    agent = create_agent(
        model=model,
        tools=tools
    )

    # --------------------------------------------------
    # Test Cases
    # --------------------------------------------------

    test_cases = [

        # Test 1
        "What is 25 plus 15?",

        # Test 2
        "What is 15 multiplied by 8?",

        # Test 3
        "What is the square root of 144?",

        # Test 4 - Multi-step calculation
        "What is the square root of the area of a rectangle "
        "with length 15 and width 8?",

        # Test 5 - Boundary case
        "What is 100 divided by 0?"
    ]


    # --------------------------------------------------
    # Execute Test Cases
    # --------------------------------------------------

    for index, question in enumerate(test_cases, start=1):

        print("\n" + "=" * 60)
        print(f"TEST CASE {index}")
        print("=" * 60)

        print(f"\nQuestion: {question}")

        try:

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

            # Get final agent message
            final_message = response["messages"][-1]

            print("\nAgent Answer:")
            print(final_message.content)

        except Exception as e:

            print("\nAgent Error:")
            print(str(e))


# --------------------------------------------------
# 5. Application Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())