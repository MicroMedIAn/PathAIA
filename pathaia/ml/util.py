# coding: utf8
"""
A module to help learners and predictors to handle files and os stuffs.

***********************************************************************
"""
import os
from skimage.io import imread
from numpy.random import shuffle


def imfiles_in_folder(folder,
                      authorized=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                      forbiden=["thumbnail"],
                      randomize=False,
                      datalim=None):
    """
    Return image paths in a given folder.

    *************************************
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
    Yield images in folder.

    ***********************
    """
    for imfile in imfiles_in_folder(folder, authorized, forbiden, randomize, datalim):
        if paths:
            yield imfile, imread(imfile)
        else:
            yield imread(imfile)


def dataset2folders(projfolder, level, randomize=False, slide_data_lim=None):
    """
    Get all slide image folders at the right level.

    ***********************************************
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
