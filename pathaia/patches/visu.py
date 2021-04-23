# coding: utf8
"""Useful functions for visualizing patches in WSIs."""

import numpy
from skimage.morphology import binary_dilation, disk
import openslide
from typing import Sequence, Tuple
from ..util.types import NDByteImage, Patch


def preview_from_queries(
    slide: openslide.OpenSlide,
    queries: Sequence[Patch],
    min_res: int = 512,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    cell_size: int = 20,
    patch_size: Tuple[int, int] = (None, None)
) -> NDByteImage:
    """
    Give thumbnail with patches displayed.

    Args:
        slide: openslide object
        queries: patch objects to preview from
        min_res: minimum size for the smallest side of the thumbnail (usually the width)
        color: rgb color for patch boundaries
        thickness: thickness of patch boundaries
        cell_size: size of a cell representing a patch in the grid

    Returns:
        Thumbnail image with patches displayed.

    """
    # get thumbnail first
    w, h = slide.dimensions
    dx, dy = patch_size
    if dx is None: dx = queries[0]["dx"]
    if dy is None: dy = queries[0]["dy"]
    thumb_w = max(512, (w // dx)*(thickness + cell_size)+thickness)
    thumb_h = max(512, (h // dy)*(thickness + cell_size)+thickness)
    image = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_w, thumb_h = image.size
    dsr_w = w / thumb_w
    dsr_h = h / thumb_h
    image = numpy.array(image)[:, :, 0:3]
    # get grid
    grid = 255 * numpy.ones((thumb_h, thumb_w), numpy.uint8)
    for query in queries:
        # position in queries are absolute
        x = int(query["x"] / dsr_w)
        y = int(query["y"] / dsr_h)
        dx = int(query["dx"] / dsr_w)
        dy = int(query["dy"] / dsr_h)
        startx = min(x, thumb_w - 1)
        starty = min(y, thumb_h - 1)
        endx = min(x + dx, thumb_w - 1)
        endy = min(y + dy, thumb_h - 1)
        # horizontal segments
        grid[starty, startx:endx] = 0
        grid[endy, startx:endx] = 0
        # vertical segments
        grid[starty:endy, startx] = 0
        grid[starty:endy, endx] = 0
    grid = grid < 255
    d = disk(thickness)
    grid = binary_dilation(grid, d)
    image[grid] = color
    return image
