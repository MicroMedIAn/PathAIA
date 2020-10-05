# coding: utf8
"""Useful functions for visualizing patches in WSIs."""

import numpy
from skimage.measure import label
from skimage.segmentation import mark_boundaries


def preview_from_queries(slide, queries, level_preview=3):
    """
    Give thumbnail with patches displayed.

    Arguments:
        - slide: openslide object

    Returns:
        - thumbnail: thumbnail image with patches displayed.

    """
    # get thumbnail first
    image = slide.read_region((0, 0), level_preview, (slide.level_dimensions[level_preview]))
    # get grid
    grid = 255 * numpy.ones((image.shape[0], image.shape[1]), numpy.uint8)
    for query in queries:
        # position in queries are absolute
        x = int(query["x"] / 2 ** query["level"])
        y = int(query["y"] / 2 ** query["level"])
        grid[y, :] = 0
        grid[:, x] = 0
    markers = label(grid)
    return mark_boundaries(image, markers)
