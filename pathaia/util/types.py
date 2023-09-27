import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy
from nptyping import NDArray, Shape
from openslide import OpenSlide
from PIL import Image

try:
    from cucim import CuImage
except ImportError:
    pass


class _CoordBase(NamedTuple):
    x: int
    y: int


class Coord(_CoordBase):
    """
    An (x, y) tuple representing integer coordinates.
    If only x is given then takes value (x, x).
    """

    def __new__(cls, x: Union[int, Iterable], y: Optional[int] = None):
        if y is None:
            if isinstance(x, dict):
                x, y = x["x"], x["y"]
            elif isinstance(x, Iterable):
                x, y = x
            else:
                y = x
        return super().__new__(cls, int(x), int(y))

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

    @classmethod
    def from_csv_row(cls, row: Dict[str, Union[str, int]], slidename: str = None):
        return cls(
            id=row["id"],
            slidename=slidename,
            position=Coord(row["x"], row["y"]),
            level=int(row["level"]),
            size_0=Coord(row["dx"], row["dy"]),
            size=Coord(row["size_x"], row["size_y"]),
        )


Filter = Sequence[Union[str, Callable]]
FilterList = Union[str, Sequence[Filter], Dict[int, Sequence[Filter]]]
PathLike = Union[str, os.PathLike]

NDByteImage = NDArray[Shape["H, W, 3"], numpy.uint8]
NDFloat32Image = NDArray[Shape["H, W, 3"], numpy.float32]
NDFloat64Image = NDArray[Shape["H, W, 3"], numpy.float64]
NDFloatImage = Union[NDFloat32Image, NDFloat64Image]
NDImage = Union[NDByteImage, NDFloatImage]

NDByteGrayImage = NDArray[Shape["H, W"], numpy.uint8]
NDFloat32GrayImage = NDArray[Shape["H, W"], numpy.float32]
NDFloat64GrayImage = NDArray[Shape["H, W"], numpy.float64]
NDFloatGrayImage = Union[NDFloat32GrayImage, NDFloat64GrayImage]
NDGrayImage = Union[NDByteGrayImage, NDFloatGrayImage]

NDBoolMask = NDArray[Shape["H, W"], numpy.bool8]
NDBoolMaskBatch = NDArray[Shape["B, H, W"], numpy.bool8]

NDIntMask2d = NDArray[Shape["H, W"], numpy.int32]
NDIntMask3d = NDArray[Shape["H, W, D"], numpy.int32]
NDIntMask4d = NDArray[Shape["H, W, D, T"], numpy.int32]

NDByteImageBatch = NDArray[Shape["B, H, W, 3"], numpy.uint8]
NDFloat32ImageBatch = NDArray[Shape["B, H, W, 3"], numpy.float32]
NDFloat64ImageBatch = NDArray[Shape["B, H, W, 3"], numpy.float64]
NDFloatImageBatch = Union[NDFloat32ImageBatch, NDFloat64ImageBatch]
NDImageBatch = Union[NDByteImageBatch, NDFloatImageBatch]

RefDataSet = Tuple[List, List]
SplitDataSet = Dict[Union[int, str], RefDataSet]
DataSet = Union[RefDataSet, SplitDataSet]


class Slide:
    def __init__(self, path: PathLike, backend: str = "openslide"):
        path = Path(path)
        if backend == "openslide":
            opener = OpenSlide
        else:
            if path.suffix not in (".svs", ".tif"):
                warnings.warn(
                    "Cucim backend only works for svs and tiff, switching to openslide."
                )
                opener = OpenSlide
                backend = "openslide"
            else:
                opener = CuImage

        self._slide = opener(str(path))
        self.backend = backend

    @property
    def dimensions(self):
        if self.backend == "openslide":
            return self._slide.dimensions
        else:
            return self._slide.size("XY")

    @property
    def _filename(self):
        if self.backend == "openslide":
            return self._slide._filename
        else:
            return self._slide.metadata["cucim"]["path"]

    def __getattr__(self, name):
        try:
            return getattr(self._slide, name)
        except AttributeError as e:
            if self.backend == "cucim" and name in self._slide.resolutions:
                return self._slide.resolutions[name]
            else:
                raise AttributeError(e)

    def get_best_level_for_downsample(self, downsample: float):
        if self.backend == "openslide":
            return self._slide.get_best_level_for_downsample(downsample)
        else:
            for i in range(1, self.level_count):
                if downsample < self.level_downsamples[i]:
                    return max(0, i - 1)
            return self.level_count - 1

    def read_region(self, location, level, size, **kwargs):
        if self.backend == "openslide":
            return self._slide.read_region(location, level, size)
        else:
            region = self._slide.read_region(
                location=location, level=level, size=size, **kwargs
            )
            return Image.fromarray(numpy.asarray(region)).convert("RGBA")

    def get_thumbnail(self, size: Coord):
        if self.backend == "openslide":
            return self._slide.get_thumbnail(size)
        else:
            dsr = max(*(dim / thumb for dim, thumb in zip(self.dimensions, size)))
            level = self.get_best_level_for_downsample(dsr)
            tile = self.read_region((0, 0), level, self.level_dimensions[level])
            # Apply on solid background
            bg_color = "#ffffff"
            thumb = Image.new("RGB", tile.size, bg_color)
            thumb.paste(tile, None, tile)
            thumb.thumbnail(size, Image.ANTIALIAS)
            return thumb
