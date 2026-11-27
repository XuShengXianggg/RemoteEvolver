import math

from ..mcpx import mcp


@mcp.tool()
def calculate_ndvi(red: float, near_infrared: float) -> float:
    """Compute NDVI = (NIR - red) / (NIR + red) from surface reflectances.

    Supply reflectances in the same units. The result is dimensionless.
    """
    if not (math.isfinite(red) and math.isfinite(near_infrared)):
        raise ValueError("Reflectances must be finite")
    if red < 0 or near_infrared < 0:
        raise ValueError("Reflectances must be nonnegative")
    if red + near_infrared == 0:
        raise ValueError("NDVI is undefined when both reflectances are zero")
    return (near_infrared - red) / (near_infrared + red)
