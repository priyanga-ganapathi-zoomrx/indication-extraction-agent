"""Calculator tools for the agent."""

from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together.

    Args:
        a: First integer to add
        b: Second integer to add

    Returns:
        int: The sum of a and b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together.

    Args:
        a: First integer to multiply
        b: Second integer to multiply

    Returns:
        int: The product of a and b
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides the first number by the second number.

    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)

    Returns:
        float: The quotient of a divided by b

    Raises:
        ValueError: If b is zero (division by zero)
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def get_calculator_tools():
    """Returns a list of calculator tools.

    Returns:
        list: List of calculator tool functions
    """
    return [add, multiply, divide]

