from typing import Any, Callable, Dict, Sequence, Union, NamedTuple, Optional, List, Tuple
from nptyping import NDArray
import os
import numpy
from dataclasses import dataclass


class _CoordBase(NamedTuple):
    x: int
    y: int


class Coord(_CoordBase):
    """
    An (x, y) tuple representing integer coordinates.
    If only x is given then takes value (x, x).
    """

    def __new__(cls, x: Union[_CoordBase, int], y: Optional[int] = None):
        if y is None:
            if isinstance(x, tuple):
                x, y = x
            else:
                y = x
        return super().__new__(cls, x, y)

    def __add__(self, other):
        x, y = self.__class__(other)
        return self.__class__(int(x + self.x), int(y + self.y))

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __sub__(self, other):
        other = self.__class__(other)
        return -other + self

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        x, y = self.__class__(other)
        return self.__class__(int(x * self.x), int(y * self.y))

    def __rmul__(self, other):
        return self * other

    def __floordiv__(self, other):
        x, y = self.__class__(other)
        return self.__class__(int(self.x / x), int(self.y / y))

    def __truediv__(self, other):
        return self // other

    def __rfloordiv__(self, other):
        x, y = self.__class__(other)
        return self.__class__(int(x / self.x), int(y / self.y))

    def __rtruediv__(self, other):
        return other // self


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

RefDataSet = Tuple[List, List]
SplitDataSet = Dict[Union[int, str], RefDataSet]
DataSet = Union[RefDataSet, SplitDataSet]
