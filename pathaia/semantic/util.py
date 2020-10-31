# coding: utf8
"""
Useful functions for semantic analysis.

***********************************************************************
"""
import numpy
import itertools


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
