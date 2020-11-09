# coding: utf8
"""Useful functions for handling patches in WSIs."""

import os
import numpy


def slides_in_folder(folder, extensions=[".mrxs"]):
    """
    Return slide files inside a folder for a given extension.

    Args:
        folder (str): absolute path to a directory containing slides.
        extension (list of str): file extensions of the slides.
    Returns:
        list of str: list of absolute path of slide files.

    """
    abspathlist = []
    for name in os.listdir(folder):

        if not name.startswith("."):
            for extension in extensions:
                if name.endswith(extension):
                    abspathlist.append(os.path.join(folder, name))
    return abspathlist


def slide_basename(slidepath):
    """
    Give the basename of a slide from its absolutepath.

    Args:
        slidepath (str): absolute path to a slide.

    Returns:
        basename (str): basename of the slide.

    """
    base = os.path.basename(slidepath)
    basename, ext = os.path.splitext(base)
    return basename


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
        list of str: absolute paths of image files in folder.

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
        numpy.random.shuffle(imfiles)
    if datalim is not None:
        imfiles = imfiles[0:datalim]

    return imfiles


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
        numpy.random.shuffle(keep)
    if slide_data_lim is not None:
        keep = keep[0:slide_data_lim]
    return {k: slide2folder[k] for k in keep}
