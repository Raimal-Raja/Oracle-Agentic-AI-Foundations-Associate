"""
LangChain Agent Under the Hood - Practical Example

This example demonstrates:
1. Loading environment variables
2. Initializing an LLM
3. Creating tools
4. Creating a ReAct-style agent
5. Invoking the agent with a multi-step question
6. Printing the final response

Question:
    What is 15 multiplied by 8 and then divided by 3?

Expected answer:
    40
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


# ============================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please add it to your .env file."
    )

print("OpenAI API key found.")


# ============================================================
# STEP 2: INITIALIZE THE MODEL
# ============================================================

model = init_chat_model(
    "gpt-5.5",
    model_provider="openai"
)


# ============================================================
# STEP 3: DEFINE TOOLS
# ============================================================

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


# ============================================================
# STEP 4: CREATE THE AGENT
# ============================================================

tools = [
    multiply,
    divide
]

agent = create_agent(
    model=model,
    tools=tools
)


# ============================================================
# STEP 5: RUN THE AGENT
# ============================================================

question = "What is 15 multiplied by 8 and then divided by 3?"

print("\nUser Question:")
print(question)

print("\nRunning agent...\n")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)


# ============================================================
# STEP 6: DISPLAY THE FINAL RESPONSE
# ============================================================

final_message = result["messages"][-1]

print("Final Answer:")
print(final_message.content)


# ============================================================
# OPTIONAL: DISPLAY THE COMPLETE MESSAGE HISTORY
# ============================================================

print("\n" + "=" * 60)
print("AGENT EXECUTION HISTORY")
print("=" * 60)

for index, message in enumerate(result["messages"], start=1):

    print(f"\nMessage {index}")
    print("-" * 40)

    print("Type:", type(message).__name__)

    if hasattr(message, "content") and message.content:
        print("Content:", message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("Tool Calls:")

        for tool_call in message.tool_calls:
            print("  Tool:", tool_call["name"])
            print("  Arguments:", tool_call["args"])
            print("  ID:", tool_call["id"])