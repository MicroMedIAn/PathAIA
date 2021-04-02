import warnings
from fastcore.basics import tuplify
from ..util.types import Coord


def convert_coords(coords):
    if isinstance(coords, dict):
        coords = Coord(**coords)
        warnings.warn(
            "Using dictionaries to represent coordinates is deprecated, its support will be dropped in future versions",
            category=DeprecationWarning,
        )
    else:
        coords = Coord(*tuplify(coords))
    return coords
