from fastmcp import FastMCP
import math


# Create MCP server
mcp = FastMCP("Math MCP Server")


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot divide by 0")

    return a / b


@mcp.tool
def square_root(a: float) -> float:
    """Calculate the square root of a number."""

    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number")

    return math.sqrt(a)


if __name__ == "__main__":
    # Start MCP server using STDIO transport
    mcp.run(transport="stdio")