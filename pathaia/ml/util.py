# coding: utf8
"""
Useful functions for machine learning models on images.

***********************************************************************
"""
import os
from skimage.io import imread
from numpy.random import shuffle
import numpy
import itertools


def imfiles_in_folder(folder,
                      authorized=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                      forbiden=["thumbnail"],
                      randomize=False,
                      datalim=None):
    """
    Get image files in a given folder.

    Get all image files (selected by file extension). You can remove terms
    from the research.

    Args:
        folder (str): absolute path to an image directory.
        authorized (list): authorized image file extensions.
        forbiden (list): non-authorized words in file names.
        randomize (bool): whether to randomize output list of files.
        datalim (int or None): maximum number of file to extract in folder.

    Returns:
        list: absolute paths of image files in folder.

    """
    imfiles = []
    for name in os.listdir(folder):
        _, ext = os.path.splitext(name)
        if ext in authorized:
            auth = True
            for forb in forbiden:
                if forb in name:
                    auth = False
            if auth:
                imfiles.append(os.path.join(folder, name))

    if randomize:
        shuffle(imfiles)
    if datalim is not None:
        imfiles = imfiles[0:datalim]

    return imfiles


def images_in_folder(folder,
                     authorized=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                     forbiden=["thumbnail"],
                     randomize=False,
                     datalim=None,
                     paths=False):
    """
    Get images in a given folder.

    Get all images as numpy arrays (selected by file extension).
    You can remove terms from the research.

    Args:
        folder (str): absolute path to an image directory.
        authorized (list): authorized image file extensions.
        forbiden (list): non-authorized words in file names.
        randomize (bool): whether to randomize output list of files.
        datalim (int or None): maximum number of file to extract in folder.
        paths (bool): whether to return absolute path with image data.

    Returns:
        iterator: yield images as numpy arrays.

    """
    for imfile in imfiles_in_folder(folder, authorized, forbiden, randomize, datalim):
        if paths:
            yield imfile, imread(imfile)
        else:
            yield imread(imfile)


def dataset2folders(projfolder, level, randomize=False, slide_data_lim=None):
    """
    Link slidenames to their pathaia patch folder.

    A pathaia patch folder is named with an int corresponding
    to the level of patch extraction.

    Args:
        projfolder (str): absolute path to a pathaia project folder.
        level (int): pyramid level of patch extraction to consider.
        randomize (bool): whether to randomize output list of slides.
        slide_data_lim (int or None): number of slides to consider in project.

    Returns:
        dict: slidenames are keys, absolute path to patch dir are values.

    """
    slide2folder = dict()
    for slidename in os.listdir(projfolder):
        slide_folder = os.path.join(projfolder, slidename)
        if os.path.isdir(slide_folder):
            level_folder = os.path.join(slide_folder, "level_{}".format(level))
            if os.path.isdir(level_folder):
                slide2folder[slidename] = level_folder

    keep = list(slide2folder.keys())
    if randomize:
        shuffle(keep)
    if slide_data_lim is not None:
        keep = keep[0:slide_data_lim]
    return {k: slide2folder[k] for k in keep}


def unlabeled_regular_grid_list(shape, step):
    """
    Get a regular grid of position on a slide given its dimensions.

    Args:
        shape (tuple): shape (i, j) of the window to tile.
        step (int): steps in pixels between patch samples.

    Returns:
        list: positions (i, j) on the regular grid.

    """
    maxi = step * int(shape[0] / step)
    maxj = step * int(shape[1] / step)
    col = numpy.arange(start=0, stop=maxj, step=step, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step, dtype=int)
    return list(itertools.product(line, col))
