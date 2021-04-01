from typing import Any, Callable, Dict, Sequence, Union, NamedTuple, Optional, List
from nptyping import NDArray
import os
import numpy
from dataclasses import dataclass


class Coord(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True)
class Patch:
    id: str
    slidename: str
    position: Coord
    level: int
    size: Coord
    size_0: Coord
    parent: Optional["Patch"] = None

    @classmethod
    def get_fields(cls) -> List[str]:
        return [
            "id",
            "global_id",
            "x",
            "y",
            "level",
            "dx",
            "dy",
            "size_x",
            "size_y",
            "parent",
        ]

    def to_csv_row(self) -> Dict[str, Union[str, int]]:
        return {
            "id": self.id,
            "global_id": self.slidename + self.id,
            "x": self.position[0],
            "y": self.position[1],
            "level": self.level,
            "dx": self.size_0[0],
            "dy": self.size_0[1],
            "size_x": self.size[0],
            "size_y": self.size[1],
            "parent": "None" if self.parent is None else self.parent.id,
        }


Filter = Sequence[Union[str, Callable]]
FilterList = Union[str, Sequence[Filter], Dict[int, Sequence[Filter]]]
PathLike = Union[str, os.PathLike]

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
