# coding: utf8
"""A module to filter patches in a slide."""
from skimage.color import rgb2lab
import numpy


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class UnknownMethodError(Error):
    """
    Raise when no class is found in a datafolder.

    *********************************************
    """

    pass


def standardize_filters(filters, top_level, low_level):
    """Check validity of hierarchical filters."""
    # check filters
    if type(filters) == str:
        level_filters = {k: [filters] for k in range(top_level, low_level - 1, -1)}
    elif type(filters) == list:
        level_filters = {k: filters for k in range(top_level, low_level - 1, -1)}
    elif type(filters) == dict:
        level_filters = dict()
        for k in range(top_level, low_level - 1, -1):
            if k in filters:
                level_filters[k] = filters[k]
            else:
                level_filters[k] = []
    else:
        raise UnknownMethodError("{} is not a valid filter !!!".format(filters))

    return level_filters


def filter_hasdapi(image, dapi_channel=0, tolerance=1):
    """
    Give presence of dapi in a patch.

    Args:
        image (ndarray): image of a patch.
        dapi_channel (int): channel to extract dapi signal.
        tolerance (int): value on dapi intensity encountered to accept a patch.
    Returns:
        bool: whether dapi is visible in slide.

    """
    return (image[:, :, dapi_channel] > tolerance).any() > 0


def filter_has_significant_dapi(image, dapi_channel=0, tolerance=0.5, dapi_tolerance=1):
    """
    Give if > 50% of dapi in a patch.

    Args:
        image (ndarray): image of a patch.
        dapi_channel (int): channel to extract dapi signal.
        tolerance (int): value on dapi intensity encountered to accept a patch.
    Returns:
        bool: whether dapi is visible in slide.

    """
    return (image[:, :, dapi_channel] > dapi_tolerance).sum() > tolerance * (image.shape[0] * image.shape[1])


def get_tissue_from_rgb(image, blacktol=0, whitetol=230):
    """
    Return the tissue mask segmentation of an image.

    True pixels for the tissue, false pixels for the background.

    Args:
        image (ndarray): image of a patch.
        blacktol (float or int): tolerance value for black pixels.
        whitetol (float or int): tolerance value for white pixels.

    Returns:
        2D-array: true pixels are tissue, false are background.

    """
    binarymask = numpy.zeros_like(image[..., 0], bool)

    for color in range(3):
        # for all color channel, find extreme values corresponding to black or white pixels
        binarymask = binarymask | ((image[..., color] < whitetol) & (image[..., color] > blacktol))

    return binarymask


def get_tissue_from_lab(image, blacktol=5, whitetol=90):
    """
    Return the tissue mask segmentation of an image.

    This version operates in the lab space, conversion of the image from
    rgb to lab is performed first.

    Args:
        image (ndarray): image of a patch.
        blacktol (float or int): tolerance value for black pixels.
        whitetol (float or int): tolerance value for white pixels.

    Returns:
        2D-array: true pixels are tissue, false are background.

    """
    image = rgb2lab(image)[..., 0]
    binarymask = numpy.ones_like(image, bool)    
    binarymask = binarymask & (image < whitetol) & (image > blacktol)
    return binarymask


def get_tissue(image, blacktol=5, whitetol=90, method="lab"):
    """
    Return the tissue mask segmentation of an image.

    One can choose the segmentation method.

    Args:
        image (ndarray): image of a patch.
        blacktol (float or int): tolerance value for black pixels.
        whitetol (float or int): tolerance value for white pixels.
        method (str): one of 'lab' or 'rgb', function to be called.

    Returns:
        2D-array: true pixels are tissue, false are background.

    """
    if method not in ["lab", "rgb"]:
        raise UnknownMethodError("Method {} is not implemented!".format(method))
    if method == "lab":
        return get_tissue_from_lab(image, blacktol, whitetol)
    if method == "rgb":
        return get_tissue_from_rgb(image, blacktol, whitetol)


def filter_has_tissue_he(image, blacktol=5, whitetol=90):
    """
    Return true if tissue inside the patch.

    This version operates in the lab space, conversion of the image from
    rgb to lab is performed first.

    Args:
        image (ndarray): image of a patch.
        blacktol (float or int): tolerance value for black pixels.
        whitetol (float or int): tolerance value for white pixels.

    Returns:
        bool: true if tissue is detected.

    """
    return get_tissue_from_lab(image, blacktol=blacktol, whitetol=whitetol).any()
