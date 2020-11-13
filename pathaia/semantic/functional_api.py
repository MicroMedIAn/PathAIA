# coding: utf8
"""
A module to produce semantic segmentations.

Can be used visualize model perception of an image.
"""
import numpy
from ..util.images import unlabeled_regular_grid_list
from ..util.paths import imfiles_in_folder
import os
import tifffile
from skimage.io import imread


def coarse(image, model):
    """Compute a coarse semantic segmentation of an image.

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
    preds = model.predict(numpy.array(patches))
    for idx, pos in enumerate(positions):
        pred = preds[idx]
        i, j = pos
        segmentation[i:i + psize, j:j + psize] = pred
    return segmentation


def coarse_sep_channels_classif(image, model):
    """Compute a coarse semantic segmentation of an image.

    Place a window on every pixel, pixel value in segmentation
    is its prediction (by a model) made on the window.

    Args:
        image (ndarray): an image.
        model (pathaia model): a ml model.

    Returns:
        list of ndarray: segmentation of the given image.

    """
    psize = model.context
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    channels = image.shape[-1]
    segmentations = numpy.zeros(img.shape, int)
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    patchlists = []
    for c in range(channels):
        patchlists.append(numpy.array([img[:, :, c][i:i + psize, j:j + psize].reshape(-1) for i, j in positions]))
    preds = model.predict(patchlists)
    for idx, pos in enumerate(positions):
        for c in range(channels):
            i, j = pos
            segmentations[i:i + psize, j:j + psize, c] = preds[c][idx]
    return segmentations


def coarse_sep_channels_desc(image, model, outrag=False):
    """Compute a coarse semantic segmentation of an image.

    Place a window on every pixel, pixel value in segmentation
    is its prediction (by a model) made on the window.

    Args:
        image (ndarray): an image.
        model (pathaia model): a ml model.

    Returns:
        list of ndarray: segmentation of the given image.

    """
    psize = model.context
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    channels = image.shape[-1]
    segmentations = numpy.zeros(img.shape + (model.n_words,), int)
    rag = numpy.zeros((img.shape[0], img.shape[1]), int)
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    patchlists = []
    for c in range(channels):
        patchlists.append(numpy.array([img[:, :, c][i:i + psize, j:j + psize].reshape(-1) for i, j in positions]))
    preds = model.predict(patchlists, fuzzy=True)
    for idx, pos in enumerate(positions):
        for c in range(channels):
            i, j = pos
            segmentations[i:i + psize, j:j + psize, c] = preds[c][idx]
            rag[i:i + psize, j:j + psize] = idx
    if outrag:
        return segmentations, rag
    return segmentations


def partition_slide_coarse(slidefolder, level, outfolder, model, sep_channels):
    """Compute a coarse semantic segmentation of all patches of a slide.

    Do the coarse image segmentation on every patch extracted at a given level.

    Args:
        slidefolder (str): absolute path to a pathaia slide folder.
        level (int): level of patches in the pyramid.
        outfolder (str): absolute path to an output folder for segmentations.
        model (pathaia model): a ml predictor.
        sep_channels (boolean): whether the model is multchannel.

    """
    imfolder = os.path.join(slidefolder, "level_{}".format(level))
    if sep_channels:
        partition_ = coarse_sep_channels_classif
    else:
        partition_ = coarse
    for imfile in imfiles_in_folder(imfolder, forbiden=["thumbnail", "semantic"]):
        img = imread(imfile)
        seg = partition_(img, model)
        imgbase = os.path.basename(imfile)
        imgbase, _ = os.path.splitext(imgbase)
        outfile = "{}_semantic.tiff".format(imgbase)
        outfile = os.path.join(outfolder, outfile)
        tifffile.imsave(outfile, seg)
