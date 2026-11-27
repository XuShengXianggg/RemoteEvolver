from ..mcpx import mcp


@mcp.tool()
def arithmetic_mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a nonempty list of numeric observations."""
    if not values:
        raise ValueError('values must not be empty')
    return sum(values) / len(values)
