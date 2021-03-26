from typing import Any, Callable, Dict, Sequence, Union
from nptyping import NDArray
import os
import numpy

Filter = Sequence[Union[str, Callable]]
FilterList = Union[str, Sequence[Filter], Dict[int, Sequence[Filter]]]
PathLike = Union[str, os.PathLike]
Patch = Dict[str, Union[str, int]]
NDImage = NDArray[(Any, Any, 3), numpy.uint8]
NDBoolMask = NDArray[(Any, Any), bool]
NDIntMask2d = NDArray[(Any, Any), int]
NDIntMask3d = NDArray[(Any, Any, Any), int]
NDIntMask4d = NDArray[(Any, Any, Any, Any), int]
