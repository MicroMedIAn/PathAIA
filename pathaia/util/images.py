# coding: utf8
"""Useful functions for images."""
import numpy
from skimage.io import imread
from .paths import imfiles_in_folder
import itertools


def regular_grid(shape, step):
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        shape (dictionary): {"x", "y"} shape of the window to tile.
        step (dictionary): {"x", "y"} steps between patch samples.

    Yields:
        dictionary: {"x", "y"} positions on a regular grid.

    """
    maxi = step["y"] * int(shape["y"] / step["y"])
    maxj = step["x"] * int(shape["x"] / step["y"])
    col = numpy.arange(start=0, stop=maxj, step=step["x"], dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step["y"], dtype=int)
    for i, j in itertools.product(line, col):
        yield {"x": j, "y": i}


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


def images_in_folder(
    folder,
    authorized=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    forbiden=("thumbnail",),
    randomize=False,
    datalim=None,
    paths=False,
):
    """
    Get images in a given folder.

    Get all images as numpy arrays (selected by file extension).
    You can remove terms from the research.

    Args:
        folder (str): absolute path to an image directory.
        authorized (list or tuple): authorized image file extensions.
        forbiden (list or tuple): non-authorized words in file names.
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


def sample_img(image, psize, spl_per_image):
    """Fit vocabulary on a single image.

    Split image in patches and fit on them.

    Args:
        image (ndarray): numpy image to fit on.
        psize (int): size in pixels of the side of a patch.
        spl_per_image (int): maximum number of patches to extract in image.

    Returns:
        list of ndarray: patches in the image.

    """
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    numpy.random.shuffle(positions)
    positions = positions[0:spl_per_image]
    patches = [img[i : i + psize, j : j + psize].reshape(-1) for i, j in positions]
    return patches


def sample_img_sep_channels(image, psize, spl_per_image):
    """Fit vocabulary on a single image.

    Split image in patches and fit on them.

    Args:
        image (ndarray): numpy image to fit on.
        psize (int): size in pixels of the side of a patch.
        spl_per_image (int): maximum number of patches to extract in image.

    Returns:
        tuple of list of ndarray: patches in the image in separated channels.

    """
    img = image.astype(float)
    n_channels = image.shape[-1]
    spaceshape = (image.shape[0], image.shape[1])
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    numpy.random.shuffle(positions)
    positions = positions[0:spl_per_image]
    patches = []
    for c in range(n_channels):
        patches.append(
            [
                img[:, :, c][i : i + psize, j : j + psize].reshape(-1)
                for i, j in positions
            ]
        )
    return tuple(patches)
