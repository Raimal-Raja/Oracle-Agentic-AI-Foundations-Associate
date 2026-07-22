"""
LangChain Building Blocks Demo

This example demonstrates:

1. Loading environment variables
2. Initializing an OpenAI chat model
3. Invoking the model directly
4. PromptTemplate
5. ChatPromptTemplate
6. LangChain Expression Language (LCEL)
7. Output parsing
8. Simple in-memory conversation context

Requirements:

    pip install -U langchain langchain-openai python-dotenv

Create a .env file:

    OPENAI_API_KEY=your_api_key_here

Never commit your .env file to GitHub.
Add .env to .gitignore.
"""

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load Environment Variables
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
# 2. Initialize the Chat Model
# ============================================================

# Change the model name if this model is not available
# in your OpenAI account.

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0,
)


# ============================================================
# 3. Invoke the Model Directly
# ============================================================

print("\n" + "=" * 60)
print("DIRECT MODEL INVOCATION")
print("=" * 60)

response = model.invoke(
    "What is LangChain?"
)

print("\nModel Response:")
print(response.content)


# ============================================================
# 4. Prompt Template
# ============================================================

print("\n" + "=" * 60)
print("PROMPT TEMPLATE")
print("=" * 60)

prompt = PromptTemplate.from_template(
    "Explain {topic} to a beginner."
)

formatted_prompt = prompt.invoke(
    {
        "topic": "AI agents"
    }
)

print("\nFormatted Prompt:")
print(formatted_prompt)


# ============================================================
# 5. Chat Prompt Template
# ============================================================

print("\n" + "=" * 60)
print("CHAT PROMPT TEMPLATE")
print("=" * 60)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful programming instructor."
        ),
        (
            "human",
            "Explain {concept} with a simple Python example."
        ),
    ]
)

chat_messages = chat_prompt.invoke(
    {
        "concept": "for loops"
    }
)

print("\nChat Messages:")

for message in chat_messages.messages:
    print(
        f"{message.type.upper()}: "
        f"{message.content}"
    )


# ============================================================
# 6. Build a Chain
# ============================================================

print("\n" + "=" * 60)
print("LANGCHAIN CHAIN")
print("=" * 60)

parser = StrOutputParser()

chain = (
    chat_prompt
    | model
    | parser
)


# ============================================================
# 7. Invoke the Chain
# ============================================================

print("\nInvoking Chain...")

chain_result = chain.invoke(
    {
        "concept": "for loops"
    }
)

print("\nChain Output:")
print(chain_result)


# ============================================================
# 8. Reuse the Same Chain with Another Input
# ============================================================

print("\n" + "=" * 60)
print("REUSING THE CHAIN")
print("=" * 60)

another_result = chain.invoke(
    {
        "concept": "AI agents"
    }
)

print("\nChain Output for AI Agents:")
print(another_result)


# ============================================================
# 9. Create Simple In-Memory Conversation History
# ============================================================

print("\n" + "=" * 60)
print("MEMORY / CONVERSATION CONTEXT")
print("=" * 60)

memory = [
    HumanMessage(
        content=(
            "My name is Alex. "
            "I am learning LangChain."
        )
    ),
    AIMessage(
        content=(
            "Nice to meet you, Alex. "
            "LangChain is a great choice for "
            "learning LLM application development."
        )
    ),
]

print("\nStored Conversation:")

for message in memory:
    print(
        f"{message.type.upper()}: "
        f"{message.content}"
    )


# ============================================================
# 10. Create a Memory-Aware Prompt
# ============================================================

memory_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant."
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),
        (
            "human",
            "{question}"
        ),
    ]
)


# ============================================================
# 11. Build Memory Chain
# ============================================================

memory_chain = (
    memory_prompt
    | model
    | parser
)


# ============================================================
# 12. Ask a Question Using Conversation Memory
# ============================================================

memory_result = memory_chain.invoke(
    {
        "history": memory,
        "question": "What was my name again?",
    }
)

print("\nMemory-Aware Model Response:")
print(memory_result)


# ============================================================
# 13. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    """
This demonstration covered the main LangChain building blocks:

1. Model
   The reasoning engine.

2. Prompt Template
   Creates reusable prompts with dynamic variables.

3. Chat Prompt Template
   Structures conversations using message roles.

4. Chain
   Connects components using the LCEL pipe operator.

5. Output Parser
   Converts model responses into convenient formats.

6. Memory
   Provides previous conversation context to the model.

The main LangChain pattern demonstrated here is:

    Prompt -> Model -> Output Parser

And for conversation-aware applications:

    Memory
       |
       v
    Prompt
       |
       v
    Model
       |
       v
    Output Parser
       |
       v
    Final Answer
"""
)