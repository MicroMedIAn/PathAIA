# coding: utf8
"""A module to filter patches in a slide."""
from skimage.color import rgb2lab
import numpy
from typing import Dict, Sequence, Union
from ..util.types import Filter, FilterList, NDByteImage, NDBoolMask


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


def standardize_filters(
    filters: FilterList,
    top_level: int,
    low_level: int,
) -> Dict[int, Sequence[Filter]]:
    """
    Check validity of hierarchical filters.

    Args:
        filters: filters to apply. Can be formatted as a single string, a list or a
            dictionary mapping a level to corresponding filters.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.

    Returns:
        Dictionnary mapping each level to corresponding filters.
    """
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


def filter_hasdapi(image: NDByteImage, dapi_channel: int = 0, tolerance: int = 1) -> bool:
    """
    Give presence of dapi in a patch.

    Args:
        image: image of a patch.
        dapi_channel: channel to extract dapi signal.
        tolerance: value on dapi intensity encountered to accept a patch.
    Returns:
        Whether dapi is visible in slide.

    """
    return (image[:, :, dapi_channel] > tolerance).any() > 0


def filter_has_significant_dapi(
    image: NDByteImage,
    dapi_channel: int = 0,
    tolerance: float = 0.5,
    dapi_tolerance: int = 1,
) -> bool:
    """
    Give if enough dapi is present in a patch.

    Args:
        image: image of a patch.
        dapi_channel: channel to extract dapi signal.
        tolerance: part of the patch that must contain dapi.
        dapi_tolerance: value on dapi intensity encountered to accept a patch.
    Returns:
        Whether dapi is significantally visible in slide.

    """
    return (image[:, :, dapi_channel] > dapi_tolerance).sum() > tolerance * (
        image.shape[0] * image.shape[1]
    )


def get_tissue_from_rgb(
    image: NDByteImage, blacktol: Union[float, int] = 0, whitetol: Union[float, int] = 230,
) -> NDBoolMask:
    """
    Return the tissue mask segmentation of an image.

    True pixels for the tissue, false pixels for the background.

    Args:
        image: image of a patch.
        blacktol: tolerance value for black pixels.
        whitetol: tolerance value for white pixels.

    Returns:
        Mask where true pixels are tissue, false are background.

    """
    binarymask = numpy.zeros_like(image[..., 0], bool)

    for color in range(3):
        # for all color channel, find extreme values corresponding to black or white pixels
        binarymask = binarymask | (
            (image[..., color] < whitetol) & (image[..., color] > blacktol)
        )

    return binarymask


def get_tissue_from_lab(
    image: NDByteImage, blacktol: Union[float, int] = 5, whitetol: Union[float, int] = 90,
) -> NDBoolMask:
    """
    Get the tissue mask segmentation of an image.

    This version operates in the lab space, conversion of the image from
    rgb to lab is performed first.

    Args:
        image: image of a patch.
        blacktol: tolerance value for black pixels.
        whitetol: tolerance value for white pixels.

    Returns:
        Mask where true pixels are tissue, false are background.

    """
    image = rgb2lab(image)[..., 0]
    binarymask = numpy.ones_like(image, bool)
    binarymask = binarymask & (image < whitetol) & (image > blacktol)
    return binarymask


def get_tissue(
    image: NDByteImage,
    blacktol: Union[float, int] = 5,
    whitetol: Union[float, int] = 90,
    method: str = "lab",
) -> NDBoolMask:
    """
    Get the tissue mask segmentation of an image.

    One can choose the segmentation method.

    Args:
        image: image of a patch.
        blacktol: tolerance value for black pixels.
        whitetol: tolerance value for white pixels.
        method: one of 'lab' or 'rgb', function to be called.

    Returns:
        Mask where true pixels are tissue, false are background.

    """
    if method not in ["lab", "rgb"]:
        raise UnknownMethodError("Method {} is not implemented!".format(method))
    if method == "lab":
        return get_tissue_from_lab(image, blacktol, whitetol)
    if method == "rgb":
        return get_tissue_from_rgb(image, blacktol, whitetol)


def filter_has_tissue_he(
    image: NDByteImage, blacktol: Union[float, int] = 5, whitetol: Union[float, int] = 90
) -> bool:
    """
    Return true if tissue inside the patch.

    Filters tissue using the l channel from lab space. Conversion of the image from rgb
    to lab is performed first.

    Args:
        image: image of a patch.
        blacktol: tolerance value for black pixels.
        whitetol: tolerance value for white pixels.

    Returns:
        True if tissue is detected.

    """
    return get_tissue_from_lab(image, blacktol=blacktol, whitetol=whitetol).any()
