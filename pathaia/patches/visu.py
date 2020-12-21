# coding: utf8
"""Useful functions for visualizing patches in WSIs."""

import numpy
from skimage.morphology import binary_dilation, disk


def preview_from_queries(
    slide, queries, level_preview=7, color=(255, 255, 0), thickness=3
):
    """
    Give thumbnail with patches displayed.

    Arguments:
        - slide: openslide object
        - queries: patch queries {"x", "y", "dx", "dy", "level"}
        - level_preview: int, pyramid level for preview thumbnail
        - color: tuple of int, rgb color for patch boundaries
        - thickness: int, thickness of patch boundaries

    Returns:
        - thumbnail: thumbnail image with patches displayed.

    """
    # get thumbnail first
    image = slide.read_region(
        (0, 0), level_preview, (slide.level_dimensions[level_preview])
    )
    image = numpy.array(image)[:, :, 0:3]
    # get grid
    grid = 255 * numpy.ones((image.shape[0], image.shape[1]), numpy.uint8)
    for query in queries:
        # position in queries are absolute
        x = int(query["x"] / 2 ** level_preview)
        y = int(query["y"] / 2 ** level_preview)
        dx = int(query["dx"] / 2 ** level_preview)
        dy = int(query["dy"] / 2 ** level_preview)
        endx = min(x + dx, image.shape[1] - 1)
        endy = min(y + dy, image.shape[0] - 1)
        # horizontal segments
        grid[y, x : endx] = 0
        grid[endy, x : endx] = 0
        # vertical segments
        grid[y : endy, x] = 0
        grid[y : endy, endx] = 0
    grid = grid < 255
    d = disk(thickness)
    grid = binary_dilation(grid, d)
    image[grid] = color
    return image
