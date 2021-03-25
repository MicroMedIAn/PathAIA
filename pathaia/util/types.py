from typing import Any, Callable, Dict, Sequence, Union
from nptyping import NDArray
import os
import openslide
import numpy

Filter = Sequence[Union[str, Callable]]
FilterList = Union[str, Sequence[Filter], Dict[int, Sequence[Filter]]]
PathLike = Union[str, os.PathLike]
Slide = openslide.OpenSlide
Patch = Dict[str, Union[str, int]]
NDImage = NDArray[(Any, Any, 3), numpy.uint8]
NDMask = NDArray[(Any, Any), bool]
