from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Math Server")


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
        raise ValueError("Cannot divide by zero.")

    return a / b


@mcp.tool
def square_root(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")

    return a ** 0.5


if __name__ == "__main__":
    # Run MCP server using STDIO transport
    mcp.run(transport="stdio")