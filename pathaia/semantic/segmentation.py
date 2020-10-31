# coding: utf8
"""
A module to produce semantic segmentations.

Can be used visualize model perception of an image.
"""
import numpy
from .util import unlabeled_regular_grid_list


def coarse(image, model):
    """
    Compute a coarse semantic segmentation of an image.

    Place a window on every pixel, pixel value in segmentation
    is its prediction (by a model) made on the window.

    Args:
        image (ndarray): an image.
        model (pathaia model): a ml model.

    Returns:
        ndarray: segmentation of the given image.

    """
    psize = model.context
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    segmentation = numpy.zeros(spaceshape, int)
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    patches = [img[i:i + psize, j:j + psize].reshape(-1) for i, j in positions]
    preds = model.predict(patches)
    for idx, pos in enumerate(positions):
        pred = preds[idx]
        i, j = pos
        segmentation[i:i + psize, j:j + psize] = pred
    return segmentation
