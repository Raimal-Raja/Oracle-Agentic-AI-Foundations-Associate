"""
Simple demonstration of Input Guardrails using the OpenAI Agents SDK.

This example is inspired by the Oracle Agentic AI Foundations course.

Requirements:
    pip install openai-agents pydantic
"""

from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
)


# ---------------------------------------------------
# Step 1: Define the Guardrail Output Model
# ---------------------------------------------------

class TopicCheck(BaseModel):
    is_on_topic: bool


# ---------------------------------------------------
# Step 2: Create a Checker Agent
# ---------------------------------------------------

history_checker = Agent(
    name="History Checker",
    instructions="""
    Determine whether the user's question is related to history.

    Return:
    - is_on_topic = True
      if the question is about history.

    - is_on_topic = False
      otherwise.
    """,
    output_type=TopicCheck,
)


# ---------------------------------------------------
# Step 3: Define the Input Guardrail
# ---------------------------------------------------

@input_guardrail
async def history_guardrail(ctx, agent, user_input):

    result = await Runner.run(
        history_checker,
        user_input,
        context=ctx.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_on_topic,
    )


# ---------------------------------------------------
# Step 4: Main Agent
# ---------------------------------------------------

history_agent = Agent(
    name="History Tutor",
    instructions="""
    Answer only history-related questions.
    """,
    input_guardrails=[history_guardrail],
)


# ---------------------------------------------------
# Example
# ---------------------------------------------------

async def main():

    response = await Runner.run(
        history_agent,
        "Who was Julius Caesar?"
    )

    print(response.final_output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())