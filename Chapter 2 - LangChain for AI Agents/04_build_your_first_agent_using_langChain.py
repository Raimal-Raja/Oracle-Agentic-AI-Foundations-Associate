'''
1 - Create a virtual environment
python -m venv langchain_env
langchain_env/Scripts/activate

2 - install dependecies
pip install langchain langchain-openai langgraph
pip install python-dotenv


3 - set your api key in a .env file
OPENAI_API_KEY=your_api_key_here
'''

from langchain.chat.models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# initialize the model
model = init_chat_model("opeani: gpt-4o")

# Define your tools
@tool
def multiply(a: float, b: float) -> float:
    """multiplies two numbers. Use for multiplication operations."""
    return a*b

@tool
def divide(a:float, b:float) -> float:
    """divides two numbers. Returns error if dividing by zero."""
    if b == 0:
        return "Error: Cannot divide by zero."
    return a/b

# create the langchain agent - that's it! Now you can use the agent to perform calculations using the defined tools.
agent = create_agent(model=model, tools=[multiply, divide])

# Run it
result = agent.invoke({"messages":[
    ("What is 10 multiplied by 5?")]
})