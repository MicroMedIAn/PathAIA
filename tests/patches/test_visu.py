from tests.helpers import FakeSlide
from pathaia.patches.visu import preview_from_queries
from pathaia.util.types import Coord
import numpy as np


def test_preview_from_queries():
    slide = FakeSlide()
    queries = []
    min_res = 512
    cell_size = 20
    thickness = 2
    color = (255, 255, 0)
    size = Coord(256)
    level = 1
    size_0 = size * slide.level_downsamples[level]
    slide_size = Coord(slide.dimensions)
    thickness = 2 * (thickness // 2) + 1
    res = slide_size / size_0 * (thickness + cell_size) + thickness
    thumb_w = max(min_res, res.x)
    thumb_h = max(min_res, res.y)
    thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    assert (
        preview_from_queries(
            slide,
            queries,
            min_res=min_res,
            color=color,
            thickness=thickness,
            cell_size=cell_size,
            size_0=size_0,
        )
        == thumb
    ).all()
