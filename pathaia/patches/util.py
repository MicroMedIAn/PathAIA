# coding: utf8
"""Useful functions for handling patches in WSIs."""

import numpy
import os
import itertools


def log_magnification(slide, level):
    """
    Give log magnification (power of 2) for a level in a slide.

    Arguments:
        - slide: an openslide object.
        - level: int, pyramid level in file.

    Returns:
        - power of 2 required to reach this level.

    """
    return int(numpy.log2(slide.level_dimensions[0][0] / slide.level_dimensions[level][0]))


def magnification(slide, level):
    """
    Give magnification level (coefficient) for a level in a slide.

    Arguments:
        - slide: an openslide object.
        - level: int, pyramid level in file.

    Returns:
        - zoom coefficient.

    """
    return int(slide.level_dimensions[0][0] / slide.level_dimensions[level][0])


def slides_in_folder(folder, extensions=[".mrxs"]):
    """
    Return slide files inside a folder for a given extension.

    Arguments:
        - folder: absolute path to a directory containing slides.
        - extension: list of str, file extensions of the slides.
    Returns:
        - abspathlist: list of absolute path of slide files.

    """
    abspathlist = []
    for name in os.listdir(folder):

        if name[0] != '.':
            for extension in extensions:
                if name.endswith(extension):
                    abspathlist.append(os.path.join(folder, name))
    return abspathlist


def slide_basename(slidepath):
    """
    Give the basename of a slide from its absolutepath.

    Arguments:
        - slidepath: str, absolute path to a slide.

    Returns:
        - basename: str, basename of the slide.

    """
    base = os.path.basename(slidepath)
    basename, ext = os.path.splitext(base)
    return basename


def regular_coords(image_dims, query):
    """
    Tile image with a regular grid.

    Arguments:
        - image_dims: dictionary, {"x", "y"} sizes of image to tile.
        - query: dictionary, {"x", "y"} sizes of tiles.
    Yields:
        - positions: dictionaries of patches.

    """
    maxi = query["y"] * int(image_dims["y"] / query["y"])
    maxj = query["x"] * int(image_dims["x"] / query["x"])
    col = numpy.arange(start=0, stop=maxj + 1, step=query["x"], dtype=int)
    line = numpy.arange(start=0, stop=maxi + 1, step=query["y"], dtype=int)
    for y, x in itertools.product(line, col):
        yield {"x": x, "y": y}


def regular_patch_queries(slide, query):
    """
    Tile image with a regular grid.

    Arguments:
        - image_dims: dictionary, {"x", "y"} sizes of image to tile.
        - query: dictionary, {"x", "y"} sizes of tiles.
    Yields:
        - request: dictionary, patch request.

    """
    mag = magnification(slide, query["level"])
    dx = query["x"]
    dy = query["y"]
    level = query["level"]
    image_dims = dict()
    image_dims["x"], image_dims["y"] = slide.level_dimensions[level]
    for coord in regular_coords(image_dims, query):
        request = dict()
        request["x"] = coord["x"] * mag
        request["y"] = coord["y"] * mag
        request["dims"] = (dx, dy)
        request["level"] = level
        yield request
