"""
Customer Support Multi-Agent System
Oracle Agentic AI Foundations Example

Requirements:
    pip install openai-agents
"""

from agents import Agent, Runner, function_tool


# ---------------------------------------------------
# Sample Database
# ---------------------------------------------------

ORDERS = {
    "ORD-001": {
        "status": "Shipped",
        "expected_delivery": "Tomorrow",
    },
    "ORD-002": {
        "status": "Delivered",
        "expected_delivery": "Completed",
    },
}


# ---------------------------------------------------
# Custom Function Tool
# ---------------------------------------------------

@function_tool
def lookup_order(order_id: str):

    return ORDERS.get(
        order_id,
        {"error": "Order not found"}
    )


# ---------------------------------------------------
# Refund Tool
# ---------------------------------------------------

@function_tool
def process_refund(order_id: str):

    return {
        "order": order_id,
        "refund_status": "Refund request submitted."
    }


# ---------------------------------------------------
# Order Status Agent
# ---------------------------------------------------

order_agent = Agent(
    name="Order Status Agent",
    instructions="""
    Help customers track their orders.
    Use the lookup_order tool.
    """,
    tools=[lookup_order],
)


# ---------------------------------------------------
# Refund Agent
# ---------------------------------------------------

refund_agent = Agent(
    name="Refund Agent",
    instructions="""
    Handle refund requests.
    Use the process_refund tool.
    """,
    tools=[process_refund],
)


# ---------------------------------------------------
# FAQ Agent
# (Normally uses OpenAI Hosted Web Search Tool)
# ---------------------------------------------------

faq_agent = Agent(
    name="FAQ Agent",
    instructions="""
    Answer general customer support questions.
    """,
)


# ---------------------------------------------------
# Triage Agent
# ---------------------------------------------------

triage_agent = Agent(
    name="Customer Support Triage",

    instructions="""
    Decide which specialist should handle
    the customer's request.

    Order questions -> Order Agent

    Refund questions -> Refund Agent

    General questions -> FAQ Agent
    """,

    handoffs=[
        order_agent,
        refund_agent,
        faq_agent,
    ],
)


# ---------------------------------------------------
# Example
# ---------------------------------------------------

async def main():

    response = await Runner.run(
        triage_agent,
        "Where is my order ORD-001?"
    )

    print(response.final_output)


if __name__ == "__main__":

    import asyncio

    asyncio.run(main())