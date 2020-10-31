# coding: utf8
"""
Useful functions for patch extraction in slides and images.

***********************************************************
"""
import numpy
import os
import itertools


def iter_patches(slide, rois):
    """
    Iterate on patches, from locations.

    Arguments:
        - slide: openslide object
        - rois: postion queries

    Yields:
        - query: position
        - patch: rgb image

    """
    for patch in rois:

        image = slide.read_region((patch["x"], patch["y"]),
                                  patch["level"],
                                  (patch["dx"], patch["dy"]))
        image = numpy.array(image)[:, :, 0:3]

        yield patch, image


def list_patches(slide, rois):
    """
    Put patches in a list, from their location.

    Arguments:
        - slide: openslide object
        - rois: position queries

    Yields:
        - queries: list of positions
        - patches: list of images

    """
    lp = []
    li = []

    for patch, img in iter_patches(slide, rois):
        lp.append(patch)
        li.append(img)
    return lp, li


def slides_in_folder(folder, slide_type=".mrxs"):
    """
    Get all slide paths inside a folder.

    Arguments:
        - folder: str, path to a directory containing slides

    Returns:
        - paths: list of str, list of files

    """
    abspathlist = []
    for name in os.listdir(folder):
        if not name.startswith("."):
            _, ext = os.path.splitext(name)
            if ext == slide_type:
                abspathlist.append(os.path.join(folder, name))
    return abspathlist


def slide_basename(slidepath):
    """
    Get basename of a slide from the path.

    Arguments:
        - slidepath: str, path to a slide file

    Returns:
        - basename: str, base name of the slide

    """
    base = os.path.basename(slidepath)
    slidebasename, _ = os.path.splitext(base)
    return slidebasename


def slide_get_basename(slide):
    """
    Get basename of a slide from the slide object.

    Arguments:
        - slide: an openslide object

    Returns:
        - basename: str, base name of the slide

    """
    slidepath = slide._filename
    base = os.path.basename(slidepath)
    slidebasename, _ = os.path.splitext(base)
    return slidebasename


def unlabeled_regular_grid(shape, step):
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        - shape: tuple, (i, j) shape of the window to tile.
        - step: int, steps between patch samples.

    Yields:
        - positions: tuples, (i, j) positions on a regular grid.

    """
    maxi = step * int(shape[0] / step)
    maxj = step * int(shape[1] / step)
    col = numpy.arange(start=0, stop=maxj, step=step, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step, dtype=int)
    for i, j in itertools.product(line, col):
        yield i, j


def unlabeled_regular_grid_list(shape, step):
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        - shape: tuple, (i, j) shape of the window to tile.
        - step: int, steps between patch samples.

    Yields:
        - positions: tuples, (i, j) positions on a regular grid.

    """
    maxi = step * int(shape[0] / step)
    maxj = step * int(shape[1] / step)
    col = numpy.arange(start=0, stop=maxj, step=step, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step, dtype=int)
    return list(itertools.product(line, col))


def regular_grid(shape, step):
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        - shape: dictionary, {"x", "y"} shape of the window to tile.
        - step: dictionary, {"x", "y"} steps between patch samples.

    Yields:
        - positions: dictionary, {"x", "y"} positions on a regular grid.

    """
    maxi = step["y"] * int(shape["y"] / step["y"])
    maxj = step["x"] * int(shape["x"] / step["y"])
    col = numpy.arange(start=0, stop=maxj, step=step["x"], dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step["y"], dtype=int)
    for i, j in itertools.product(line, col):
        yield {"x": j, "y": i}
