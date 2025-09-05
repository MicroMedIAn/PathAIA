# coding: utf8
"""Useful functions for images."""
import numpy
from skimage.io import imread
from skimage.transform import resize
from .paths import imfiles_in_folder
from .types import NDBoolMask, PathLike, NDImage, NDByteImage, Coord
from ..patches.compat import convert_coords
import itertools
from typing import Iterator, List, Tuple, Sequence, Optional, Union, Any
from nptyping import NDArray, Shape, Float


def regular_grid(shape: Coord, interval: Coord, psize: Coord) -> Iterator[Coord]:
    """
    Get a regular grid of position on a slide given its dimensions.

    Arguments:
        shape: (x, y) shape of the window to tile.
        interval: (x, y) steps between patch samples.
        psize: (w, h) size of the patches (in pixels).

    Yields:
        (x, y) positions on a regular grid.

    """
    psize = convert_coords(psize)
    interval = convert_coords(interval)
    shape = convert_coords(shape)
    step = interval + psize
    maxj, maxi = step * ((shape - psize) / step + 1)
    col = numpy.arange(start=0, stop=maxj, step=step[0], dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step[1], dtype=int)
    for i, j in itertools.product(line, col):
        yield Coord(x=j, y=i)


def get_coords_from_mask(
    mask: NDBoolMask, shape: Coord, interval: Coord, psize: Coord
) -> Iterator[Coord]:
    """
    Get tissue coordinates given a tissue binary mask and slide dimensions.

    Arguments:
        mask: binary mask where tissue is marked as True.
        shape: (x, y) shape of the window to tile.
        interval: (x, y) steps between patch samples.
        psize: (w, h) size of the patches (in pixels).

    Yields:
        (x, y) positions on a regular grid.
    """

    psize = convert_coords(psize)
    interval = convert_coords(interval)
    shape = convert_coords(shape)
    step = interval + psize
    mask_w, mask_h = (shape - psize) / step + 1
    mask = resize(mask, (mask_h, mask_w))
    for i, j in numpy.argwhere(mask):
        yield step * (j, i)


def unlabeled_regular_grid_list(shape: Coord, step: int, psize: int) -> List[Coord]:
    """
    Get a regular grid of position on a slide given its dimensions.

    Args:
        shape: shape (i, j) of the window to tile.
        step: steps in pixels between patch samples.
        psize: size of the side of the patch (in pixels).

    Returns:
        Positions (i, j) on the regular grid.

    """
    maxi = step * int((shape[0] - (psize - step)) / step) + 1
    maxj = step * int((shape[1] - (psize - step)) / step) + 1
    col = numpy.arange(start=0, stop=maxj, step=step, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=step, dtype=int)
    return list(itertools.product(line, col))


def images_in_folder(
    folder: PathLike,
    authorized: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    forbiden: Sequence[str] = ("thumbnail",),
    randomize: bool = False,
    datalim: Optional[int] = None,
    paths: bool = False,
) -> Iterator[Union[NDByteImage, Tuple[str, NDByteImage]]]:
    """
    Get images in a given folder.

    Get all images as numpy arrays (selected by file extension).
    You can remove terms from the research.

    Args:
        folder: absolute path to an image directory.
        authorized: authorized image file extensions.
        forbiden: non-authorized words in file names.
        randomize: whether to randomize output list of files.
        datalim: maximum number of file to extract in folder.
        paths: whether to return absolute path with image data.

    Yields:
        Images as numpy arrays, optionally with path.

    """
    for imfile in imfiles_in_folder(folder, authorized, forbiden, randomize, datalim):
        if paths:
            yield imfile, imread(imfile)
        else:
            yield imread(imfile)


def sample_img(
    image: NDImage, psize: int, spl_per_image: int, mask: NDBoolMask = None
) -> List[NDArray[Shape["N"], Float]]:
    """
    Split image in patches.

    Args:
        image: numpy image to fit on.
        psize: size in pixels of the side of a patch.
        spl_per_image: maximum number of patches to extract in image.
        mask: optional boolean array, we sample in true pixels if provided.

    Returns:
        Patches in the image.

    """
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    di, dj = spaceshape
    if mask is None:
        positions = unlabeled_regular_grid_list(spaceshape, psize)
    else:
        half_size = int(0.5 * psize)
        cropped_mask = numpy.zeros_like(mask)
        cropped_mask[mask > 0] = 1
        cropped_mask[0 : half_size + 1, :] = 0
        cropped_mask[di - half_size - 1 : :, :] = 0
        cropped_mask[:, 0 : half_size + 1] = 0
        cropped_mask[:, dj - half_size - 1 : :] = 0
        y, x = numpy.where(cropped_mask > 0)
        y -= half_size
        x -= half_size
        positions = [(i, j) for i, j in zip(y, x)]

    numpy.random.shuffle(positions)
    positions = positions[0:spl_per_image]
    patches = [img[i : i + psize, j : j + psize].reshape(-1) for i, j in positions]
    return patches


def sample_img_sep_channels(
    image: NDByteImage, psize: int, spl_per_image: int, mask: NDBoolMask = None
) -> Tuple[List[NDArray[Shape["N"], Float]], ...]:
    """Fit vocabulary on a single image.

    Split image in patches and fit on them.

    Args:
        image: numpy image to fit on.
        psize: size in pixels of the side of a patch.
        spl_per_image: maximum number of patches to extract in image.
        mask: optional boolean array, we sample in true pixels if provided.

    Returns:
        Patches in the image in separated channels.

    """
    img = image.astype(float)
    n_channels = image.shape[-1]
    spaceshape = (image.shape[0], image.shape[1])
    di, dj = spaceshape
    if mask is None:
        positions = unlabeled_regular_grid_list(spaceshape, psize)
    else:
        half_size = int(0.5 * psize)
        cropped_mask = numpy.zeros_like(mask)
        cropped_mask[mask > 0] = 1
        cropped_mask[0 : half_size + 1, :] = 0
        cropped_mask[di - half_size - 1 : :, :] = 0
        cropped_mask[:, 0 : half_size + 1] = 0
        cropped_mask[:, dj - half_size - 1 : :] = 0
        y, x = numpy.where(cropped_mask > 0)
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
