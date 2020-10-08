# coding: utf8
"""A module to filter patches in a slide."""


def filter_hasdapi(image, dapi_channel=0, tolerance=1):
    """
    Give presence of dapi in a patch.

    Arguments:
        - image: image of a patch.
        - dapi_channel: int, channel to extract dapi signal.
        - tolerance: int, value on dapi intensity encountered to accept a patch.
    Returns:
        - has_dapi: bool, whether dapi is visible in slide.

    """
    return (image[:, :, dapi_channel] > tolerance).sum() > 0


def filter_has_significant_dapi(image, dapi_channel=0, tolerance=0.5, dapi_tolerance=1):
    """
    Give if > 50% of dapi in a patch.

    Arguments:
        - image: image of a patch.
        - dapi_channel: int, channel to extract dapi signal.
        - tolerance: int, value on dapi intensity encountered to accept a patch.
    Returns:
        - has_dapi: bool, whether dapi is visible in slide.

    """
    return (image[:, :, dapi_channel] > dapi_tolerance).sum() > tolerance * (image.shape[0] * image.shape[1])
