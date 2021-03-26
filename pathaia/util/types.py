from typing import Any, Callable, Dict, Sequence, Union
from nptyping import NDArray
import os
import numpy

Filter = Sequence[Union[str, Callable]]
FilterList = Union[str, Sequence[Filter], Dict[int, Sequence[Filter]]]
PathLike = Union[str, os.PathLike]
Patch = Dict[str, Union[str, int]]

NDByteImage = NDArray[(Any, Any, 3), numpy.uint8]
NDFloat32Image = NDArray[(Any, Any, 3), numpy.float32]
NDFloat64Image = NDArray[(Any, Any, 3), numpy.float64]
NDFloatImage = Union[NDFloat32Image, NDFloat64Image]
NDImage = Union[NDByteImage, NDFloatImage]

NDByteGrayImage = NDArray[(Any, Any), numpy.uint8]
NDFloat32GrayImage = NDArray[(Any, Any), numpy.float32]
NDFloat64GrayImage = NDArray[(Any, Any), numpy.float64]
NDFloatGrayImage = Union[NDFloat32GrayImage, NDFloat64GrayImage]
NDGrayImage = Union[NDByteGrayImage, NDFloatGrayImage]

NDBoolMask = NDArray[(Any, Any), bool]
NDBoolMaskBatch = NDArray[(Any, Any, Any), bool]

NDIntMask2d = NDArray[(Any, Any), int]
NDIntMask3d = NDArray[(Any, Any, Any), int]
NDIntMask4d = NDArray[(Any, Any, Any, Any), int]

NDByteImageBatch = NDArray[(Any, Any, Any, 3), numpy.uint8]
NDFloat32ImageBatch = NDArray[(Any, Any, Any, 3), numpy.float32]
NDFloat64ImageBatch = NDArray[(Any, Any, Any, 3), numpy.float64]
NDFloatImageBatch = Union[NDFloat32ImageBatch, NDFloat64ImageBatch]
NDImageBatch = Union[NDByteImageBatch, NDFloatImageBatch]
