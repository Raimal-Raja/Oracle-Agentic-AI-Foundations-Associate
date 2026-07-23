"""
LangChain Agent Under the Hood
==============================

This practical demonstrates the internal flow of a tool-using AI agent.

Question:
    What is 15 multiplied by 8 and then divided by 3?

Conceptual flow:

    User Question
          ↓
    LLM decides to call multiply
          ↓
    LangChain executes multiply()
          ↓
    Result = 120
          ↓
    LLM receives result
          ↓
    LLM decides to call divide
          ↓
    LangChain executes divide()
          ↓
    Result = 40
          ↓
    LLM returns final answer

Install:

    pip install -U langchain langchain-openai python-dotenv

Create .env:

    OPENAI_API_KEY=your_api_key_here
"""

import json
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
        "OPENAI_API_KEY is missing. "
        "Add it to your .env file."
    )

print("OpenAI API Key Found")


# ============================================================
# 2. INITIALIZE THE MODEL
# ============================================================

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0
)


# ============================================================
# 3. DEFINE TOOLS
# ============================================================

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    Use this tool when multiplication is required.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second number.
    Use this tool when division is required.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


# ============================================================
# 4. CREATE TOOL REGISTRY
# ============================================================

# Conceptually, LangChain maintains a mapping between
# tool names and Python functions.

tool_registry = {
    "multiply": multiply,
    "divide": divide,
}


print("\nTool Registry")
print("-" * 50)

for tool_name in tool_registry:
    print(f"{tool_name} -> {tool_registry[tool_name]}")


# ============================================================
# 5. DISPLAY TOOL SCHEMAS
# ============================================================

print("\nTool Schemas")
print("-" * 50)

for tool in [multiply, divide]:

    print(f"\nTool Name: {tool.name}")
    print(f"Description: {tool.description}")

    # LangChain exposes the structured input schema.
    print("Input Schema:")

    try:
        schema = tool.args_schema.model_json_schema()
        print(
            json.dumps(
                schema,
                indent=2
            )
        )

    except Exception as error:
        print(
            "Could not display schema:",
            error
        )


# ============================================================
# 6. CREATE THE AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=[
        multiply,
        divide,
    ],
)


# ============================================================
# 7. USER QUESTION
# ============================================================

question = (
    "What is 15 multiplied by 8 "
    "and then divided by 3?"
)


print("\n" + "=" * 70)
print("USER QUESTION")
print("=" * 70)

print(question)


# ============================================================
# 8. INVOKE THE AGENT
# ============================================================

print("\n" + "=" * 70)
print("STARTING AGENT")
print("=" * 70)

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


# ============================================================
# 9. DISPLAY ALL MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("AGENT EXECUTION TRACE")
print("=" * 70)


for index, message in enumerate(
    result["messages"],
    start=1
):

    print(
        f"\n--- MESSAGE {index} ---"
    )

    print(
        "Message Type:",
        message.type
    )

    # --------------------------------------------------------
    # Display message content
    # --------------------------------------------------------

    if hasattr(
        message,
        "content"
    ):

        if message.content:

            print(
                "Content:",
                message.content
            )

    # --------------------------------------------------------
    # Display tool calls
    # --------------------------------------------------------

    if hasattr(
        message,
        "tool_calls"
    ):

        if message.tool_calls:

            print(
                "\nTool Calls:"
            )

            for tool_call in message.tool_calls:

                print(
                    "Tool Name:",
                    tool_call["name"]
                )

                print(
                    "Arguments:",
                    tool_call["args"]
                )

                print(
                    "Tool Call ID:",
                    tool_call["id"]
                )


# ============================================================
# 10. DISPLAY FINAL ANSWER
# ============================================================

final_message = result["messages"][-1]


print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(
    final_message.content
)


# ============================================================
# 11. MANUAL UNDER-THE-HOOD DEMONSTRATION
# ============================================================

print("\n" + "=" * 70)
print("MANUAL TOOL-CALL FLOW")
print("=" * 70)


# ------------------------------------------------------------
# Step 1: Simulate model tool-call request
# ------------------------------------------------------------

tool_call_request = {
    "id": "call_001",
    "name": "multiply",
    "arguments": {
        "a": 15,
        "b": 8,
    },
}


print("\nStep 1: Model requests a tool call")

print(
    json.dumps(
        tool_call_request,
        indent=2
    )
)


# ------------------------------------------------------------
# Step 2: Extract tool name
# ------------------------------------------------------------

tool_name = tool_call_request["name"]

arguments = tool_call_request["arguments"]

call_id = tool_call_request["id"]


print("\nStep 2: LangChain parses the request")

print(
    "Tool:",
    tool_name
)

print(
    "Arguments:",
    arguments
)

print(
    "Call ID:",
    call_id
)


# ------------------------------------------------------------
# Step 3: Find Python function
# ------------------------------------------------------------

python_function = tool_registry.get(
    tool_name
)


print(
    "\nStep 3: Tool Registry Lookup"
)

print(
    "Python Function:",
    python_function
)


# ------------------------------------------------------------
# Step 4: Execute Python function
# ------------------------------------------------------------

tool_result = python_function.invoke(
    arguments
)


print(
    "\nStep 4: Execute Tool"
)

print(
    "Tool Result:",
    tool_result
)


# ------------------------------------------------------------
# Step 5: Create tool result message
# ------------------------------------------------------------

tool_message = {
    "role": "tool",
    "tool_call_id": call_id,
    "content": str(tool_result),
}


print(
    "\nStep 5: Tool Result Message"
)

print(
    json.dumps(
        tool_message,
        indent=2
    )
)


# ============================================================
# 12. SECOND TOOL CALL
# ============================================================

second_tool_call = {
    "id": "call_002",
    "name": "divide",
    "arguments": {
        "a": 120,
        "b": 3,
    },
}


print("\n" + "=" * 70)
print("SECOND TOOL CALL")
print("=" * 70)


print(
    json.dumps(
        second_tool_call,
        indent=2
    )
)


# ------------------------------------------------------------
# Look up second tool
# ------------------------------------------------------------

second_tool = tool_registry.get(
    second_tool_call["name"]
)


# ------------------------------------------------------------
# Execute second tool
# ------------------------------------------------------------

second_result = second_tool.invoke(
    second_tool_call["arguments"]
)


print(
    "\nSecond Tool Result:",
    second_result
)


# ============================================================
# 13. FINAL CONCEPTUAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE FLOW")
print("=" * 70)

print(
    """
User Question:
    What is 15 * 8 / 3?

        ↓

LLM Tool Call:
    multiply(15, 8)

        ↓

Python Execution:
    15 * 8

        ↓

Tool Result:
    120

        ↓

LLM Tool Call:
    divide(120, 3)

        ↓

Python Execution:
    120 / 3

        ↓

Tool Result:
    40

        ↓

Final Answer:
    40
"""
)