# coding: utf8
"""Useful functions for images."""
import numpy
from skimage.io import imread
from skimage.transform import resize
from .paths import imfiles_in_folder
import itertools


def regular_grid(shape, step, psize):
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        shape (dictionary): {"x", "y"} shape of the window to tile.
        step (dictionary): {"x", "y"} steps between patch samples.
        psize (int): size of the side of the patch (in pixels).

    Yields:
        dictionary: {"x", "y"} positions on a regular grid.

    """
    maxi = step["y"] * int((shape["y"] - (psize - step["y"])) / step["y"]) + 1
    maxj = step["x"] * int((shape["x"] - (psize - step["x"])) / step["x"]) + 1
    col = numpy.arange(start=0, stop=maxj, step=step["x"], dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step["y"], dtype=int)
    for i, j in itertools.product(line, col):
        yield {"x": j, "y": i}


def get_coords_from_mask(mask, shape, step, psize):
    """
    Get tissue coordinates given a tissue binary mask and slide dimensions.

    Arguments:
        mask (ndarray): binary mask where tissue is marked as True.
        shape (dictionary): {"x", "y"} shape of the window to tile.
        step (dictionary): {"x", "y"} steps between patch samples.
        psize (int): size of the side of the patch (in pixels).

    Yields:
        dictionary: {"x", "y"} positions on a regular grid.
    """

    mask_h = int((shape["y"] - psize) / step["y"]) + 1
    mask_w = int((shape["x"] - psize) / step["x"]) + 1
    mask = resize(mask, (mask_h, mask_w))
    for i, j in numpy.argwhere(mask):
        yield {"x": j * step["x"], "y": i * step["y"]}


def unlabeled_regular_grid_list(shape, step, psize):
    """
    Get a regular grid of position on a slide given its dimensions.

    Args:
        shape (tuple): shape (i, j) of the window to tile.
        step (int): steps in pixels between patch samples.
        psize (int): size of the side of the patch (in pixels).

    Returns:
        list: positions (i, j) on the regular grid.

    """
    maxi = step * int((shape[0] - (psize - step)) / step) + 1
    maxj = step * int((shape[1] - (psize - step)) / step) + 1
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


def sample_img(image, psize, spl_per_image, mask=None):
    """Get patches in an image.

    Split image in patches.

    Args:
        image (ndarray): numpy image to fit on.
        psize (int): size in pixels of the side of a patch.
        spl_per_image (int): maximum number of patches to extract in image.
        mask (ndarray): optional boolean array, we sample in true pixels if provided.

    Returns:
        list of ndarray: patches in the image.

    """
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    di, dj = spaceshape
    if mask is None:
        positions = unlabeled_regular_grid_list(spaceshape, psize)
    else:
        half_size = int(0.5 * psize)
        croped_mask = numpy.zeros_like(mask)
        croped_mask[mask > 0] = 1
        croped_mask[0 : half_size + 1, :] = 0
        croped_mask[di - half_size - 1 : :, :] = 0
        croped_mask[:, 0 : half_size + 1] = 0
        croped_mask[:, dj - half_size - 1 : :] = 0
        y, x = numpy.where(croped_mask > 0)
        y -= half_size
        x -= half_size
        positions = [(i, j) for i, j in zip(y, x)]

    numpy.random.shuffle(positions)
    positions = positions[0:spl_per_image]
    patches = [img[i : i + psize, j : j + psize].reshape(-1) for i, j in positions]
    return patches


def sample_img_sep_channels(image, psize, spl_per_image, mask=None):
    """Fit vocabulary on a single image.

    Split image in patches and fit on them.

    Args:
        image (ndarray): numpy image to fit on.
        psize (int): size in pixels of the side of a patch.
        spl_per_image (int): maximum number of patches to extract in image.
        mask (ndarray): optional boolean array, we sample in true pixels if provided.

    Returns:
        tuple of list of ndarray: patches in the image in separated channels.

    """
    img = image.astype(float)
    n_channels = image.shape[-1]
    spaceshape = (image.shape[0], image.shape[1])
    di, dj = spaceshape
    if mask is None:
        positions = unlabeled_regular_grid_list(spaceshape, psize)
    else:
        half_size = int(0.5 * psize)
        croped_mask = numpy.zeros_like(mask)
        croped_mask[mask > 0] = 1
        croped_mask[0 : half_size + 1, :] = 0
        croped_mask[di - half_size - 1 : :, :] = 0
        croped_mask[:, 0 : half_size + 1] = 0
        croped_mask[:, dj - half_size - 1 : :] = 0
        y, x = numpy.where(croped_mask > 0)
        y -= half_size
        x -= half_size
        positions = [(i, j) for i, j in zip(y, x)]
    numpy.random.shuffle(positions)
    if len(positions) > spl_per_image:
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


if __name__ == "__main__":
    shape = {"x": 50000, "y": 10000}
    step = {"x": 768, "y": 768}
    psize = 1024
    mask = numpy.ones((1024, 512), dtype=bool)
    old_coords = list(regular_grid(shape, step, psize))
    old_coords.sort(key=lambda x: (x["x"], x["y"]))
    new_coords = list(get_coords_from_mask(mask, shape, step, psize))
    new_coords.sort(key=lambda x: (x["x"], x["y"]))
    assert all(
        [x0 == x1 and y0 == y1 for (x0, y0), (x1, y1) in zip(old_coords, new_coords)]
    )
