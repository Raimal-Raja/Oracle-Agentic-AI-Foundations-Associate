"""
First LangChain AI Agent
------------------------
This example demonstrates a simple ReAct-style AI agent
using LangChain and OpenAI.

The agent has three tools:
1. add
2. multiply
3. divide

The agent dynamically decides which tools to use and
in what order.

Prerequisites:
    pip install -U langchain langchain-openai python-dotenv

Create a .env file:
    OPENAI_API_KEY=your_api_key_here
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please add it to your .env file."
    )

print("OpenAI API Key Found")


# ============================================================
# 2. INITIALIZE THE MODEL
# ============================================================

# The LLM acts as the "brain" of the agent.
# Replace the model name with a currently available
# model supported by your OpenAI account if necessary.

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0
)


# ============================================================
# 3. DEFINE TOOLS
# ============================================================

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    Use this tool when the user asks for an addition operation.
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    Use this tool when the user asks for multiplication.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second number.
    Use this tool when the user asks for division.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


# Put all tools into a list.
tools = [
    add,
    multiply,
    divide,
]


# ============================================================
# 4. CREATE THE AGENT
# ============================================================

# LangChain creates the agent using:
# - The LLM
# - The available tools
#
# The agent can dynamically decide:
# - Which tool to use
# - When to use it
# - Whether another tool call is required

agent = create_agent(
    model=model,
    tools=tools,
)


# ============================================================
# 5. FUNCTION TO RUN THE AGENT
# ============================================================

def run_agent(question: str):
    """
    Send a question to the AI agent and print the final answer.
    """

    print("\n" + "=" * 70)
    print("USER QUESTION")
    print("=" * 70)
    print(question)

    print("\nRunning agent...")

    # Invoke the agent.
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    # The final message is normally the last message
    # in the agent's message history.
    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_message.content)

    return result


# ============================================================
# 6. EXAMPLE 1 - SIMPLE ADDITION
# ============================================================

run_agent(
    "What is 42 plus 58?"
)


# ============================================================
# 7. EXAMPLE 2 - MULTIPLE TOOL CALLS
# ============================================================

run_agent(
    "What is 15 multiplied by 8 and then divided by 3?"
)


# ============================================================
# 8. EXAMPLE 3 - ANOTHER MULTI-STEP QUESTION
# ============================================================

run_agent(
    "Calculate 100 divided by 5 and then multiply the result by 6."
)


# ============================================================
# 9. OPTIONAL: PRINT THE FULL AGENT TRACE
# ============================================================

print("\n" + "=" * 70)
print("FULL AGENT EXECUTION TRACE")
print("=" * 70)

result = run_agent(
    "What is 15 multiplied by 8 and then divided by 3?"
)

print("\n" + "=" * 70)
print("MESSAGE TRACE")
print("=" * 70)

for index, message in enumerate(result["messages"], start=1):

    print(f"\n--- Message {index} ---")

    print("Type:", message.type)

    if hasattr(message, "content"):
        print("Content:", message.content)

    # Check whether this message contains tool calls.
    if hasattr(message, "tool_calls") and message.tool_calls:
        print("Tool Calls:")

        for tool_call in message.tool_calls:
            print("  Tool:", tool_call["name"])
            print("  Arguments:", tool_call["args"])