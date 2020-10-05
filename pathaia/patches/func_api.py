# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""
import numpy
import openslide
from skimage.morphology import dilation, erosion
from skimage.morphology import disk, square
from skimage.color import rgb2lab
import itertools
from .util import regular_grid, magnification


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


def get_tissue_from_rgb(image, blacktol=0, whitetol=230):
    """
    Return the tissue mask segmentation of an image.

    True pixels for the tissue, false pixels for the background.

    Arguments:
        - image: numpy ndarray, rgb image.
        - blacktol: float or int, tolerance value for black pixels.
        - whitetol: float or int, tolerance value for white pixels.

    Returns:
        - binarymask: true pixels are tissue, false are background.

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

    Arguments:
        - image: numpy ndarray, rgb image.
        - blacktol: float or int, tolerance value for black pixels.
        - whitetol: float or int, tolerance value for white pixels.

    Returns:
        - binarymask: true pixels are tissue, false are background.

    """
    binarymask = numpy.ones_like(image[..., 0], bool)
    image = rgb2lab(image)
    binarymask = binarymask & (image[..., 0] < whitetol) & (image[..., 0] > blacktol)
    return binarymask


def get_tissue(image, blacktol=5, whitetol=90, method="lab"):
    """
    Return the tissue mask segmentation of an image.

    One can choose the segmentation method.

    Arguments:
        - image: numpy ndarray, rgb image.
        - blacktol: float or int, tolerance value for black pixels.
        - whitetol: float or int, tolerance value for white pixels.
        - method: str, one of 'lab' or 'rgb', function to be called.

    Returns:
        - binarymask: true pixels are tissue, false are background.

    """
    if method not in ["lab", "rgb"]:
        raise UnknownMethodError("Method {} is not implemented!".format(method))
    if method == "lab":
        return get_tissue_from_lab(image, blacktol, whitetol)
    if method == "rgb":
        return get_tissue_from_rgb(image, blacktol, whitetol)


def slide_rois(slide, level, psize, interval, offsetx=0, offsety=0, coords=True, tissue=True):
    """
    Return the absolute coordinates of patches.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels.

    Arguments:
        - slide: openslide object.
        - level: int, pyramid level.
        - psize: int
        - interval: interval between 2 neighboring patches.
        - offsetx: int, inf to psize, offset on x axis for patch start.
        - offsety: int, inf to psize, offset on y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.
        - tissue: bool, only images > 50% tissue will be yielded if set to True.

    Yields:
        - image: numpy array rgb image.
        - coords: tuple of numpy arrays, (icoords, jcoords).

    """
    if tissue:
        for patch in slide_rois_tissue_(slide, level, psize, interval, offsetx, offsety, coords):
            yield patch
    else:
        for patch in slide_rois_(slide, level, psize, interval, offsetx, offsety, coords):
            yield patch


def slide_rois_(slide, level, psize, interval, offsetx, offsety, coords):
    """
    Return the absolute coordinates of patches.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels.

    Arguments:
        - slide: openslide object.
        - level: int, pyramid level.
        - psize: int
        - interval: interval between 2 neighboring patches.
        - offsetx: int, inf to psize, offset on x axis for patch start.
        - offsety: int, inf to psize, offset on y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.

    Yields:
        - image: numpy array rgb image.
        - coords: tuple of numpy arrays, (icoords, jcoords).

    """
    dim = slide.level_dimensions[level]
    mag = magnification(slide, level)
    for i, j in regular_grid((dim[1], dim[0]), interval):
        y = i * mag + offsety
        x = j * mag + offsetx
        try:
            image = slide.read_region((x, y), level, (psize, psize))
            image = numpy.array(image)[:, :, 0:3]
            if coords:
                yield image, (x, y)
            else:
                yield image
        except openslide.lowlevel.OpenSlideError:
            print("small failure while reading tile ", x, ' ', y, ' in ', slide._filename)


def slide_rois_tissue_(slide, level, psize, interval, offsetx, offsety, coords):
    """
    Return the absolute coordinates of patches.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels.

    Arguments:
        - slide: openslide object.
        - level: int, pyramid level.
        - psize: int
        - interval: interval between 2 neighboring patches.
        - offsetx: int, inf to psize, offset on x axis for patch start.
        - offsety: int, inf to psize, offset on y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.

    Yields:
        - image: numpy array rgb image.
        - coords: tuple of numpy arrays, (icoords, jcoords).

    """
    dim = slide.level_dimensions[level]
    mag = magnification(slide, level)
    for i, j in regular_grid((dim[1], dim[0]), interval):
        y = i * mag + offsety
        x = j * mag + offsetx
        try:
            image = slide.read_region((x, y), level, (psize, psize))
            image = numpy.array(image)[:, :, 0:3]
            if get_tissue(image).sum() > 0.5 * psize * psize:
                if coords:
                    yield image, (x, y)
                else:
                    yield image
        except openslide.lowlevel.OpenSlideError:
            print("small failure while reading tile ", x, ' ', y, ' in ', slide._filename)


def gen_patch_coords(shape, psize, offseti=0, offsetj=0):
    """
    Return the coordinates of patches.

    Given a shape, a patchsize in pixels and an offset in pixels.

    Arguments:
        - shape: tuple of int shape of the image to patchify.
        - psize: int, size in pixel of patch side.
        - offseti: int, inf to psize, offset on lines for patch start.
        - offsetj: int, inf to psize, offset on columns for patch start.

    Returns:
        - coords: tuple of numpy arrays, (icoords, jcoords).

    """
    maxi = max([offseti + psize, psize * int(shape[0] / psize)])
    maxj = max([offsetj + psize, psize * int(shape[1] / psize)])
    # print('gen_patch_coords: ', maxi, maxj)
    col = numpy.arange(start=offsetj, stop=maxj, step=psize, dtype=int)
    line = numpy.arange(start=offseti, stop=maxi, step=psize, dtype=int)
    i = []
    j = []

    for p in itertools.product(line, col):
        i.append(p[0])
        j.append(p[1])

    return numpy.array(i), numpy.array(j)


def get_patch_children(node):
    """
    Return the children patches.

    Given a patch.
    """
    x, y, level, size = node

    if level == 0:

        return node

    children_level = level - 1
    abs_size = size * (2 ** level)
    children_abs_size = int(abs_size / 2)

    return [(x, y, children_level, size),
            (x + children_abs_size, y, children_level, size),
            (x, y + children_abs_size, children_level, size),
            (x + children_abs_size, y + children_abs_size, children_level, size)]


def naive_patch_placement_optimization(mask, psize, verbose=False):
    """
    Find optimal non-overlapping patch placement.

    The one that have the bigest
    intersection with the mask, given a mask and a patchsize in pixels.

    Arguments:
        - mask: numpy ndarray, binary mask of tissue.
        - psize: patch side size in pixels.

    Returns:
        - coordinates: tuple of numpy arrays (icoords, jcoords).

    """
    offsets = itertools.product(list(range(psize)), list(range(psize)))
    placements = [gen_patch_coords(mask.shape, psize, o[0], o[1]) for o in offsets]
    scores = []
    dilated = dilation(mask, selem=disk(16))
    eroded = erosion(dilated, selem=square(int(0.25 * psize)))

    # debug...
    displaymask = dilated.astype(int) + mask.astype(int) + eroded.astype(int)
    patchposmask = numpy.zeros_like(eroded)

    for p in placements:
        i, j = p
        # print(i.dtype, j.dtype)
        # print(i[0], j[0])
        pmask = numpy.zeros_like(mask)
        pmask[i, :] = True
        pmask[:, j] = True
        scores.append(numpy.logical_and(pmask, eroded).sum())

    scores = numpy.array(scores)

    # debug...
    # print(scores.min(), scores.max())
    idx = numpy.argmax(scores)

    # filter patch position with mask intersection
    finali = []
    finalj = []

    posi, posj = placements[idx]
    for i, j in zip(posi, posj):
        if eroded[i, j]:
            # debug
            patchposmask[i, :] = 1
            patchposmask[:, j] = 1

            finali.append(i)
            finalj.append(j)

    # debug
    if verbose:
        from matplotlib import pyplot as plt
        patchposmask = numpy.logical_and(patchposmask, eroded)
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(displaymask + dilation(patchposmask).astype(int))
        plt.show()

    return numpy.array(finali), numpy.array(finalj)


def tile_at_level(slide, patchsize, level2tile, verbose=False):
    """
    Return the placement of patches.

    Given a slide, a patchsize, a level max and a level in resolution pyramid.

    Arguments:
        - slide: OpenSlide object.
        - patchsize: int, size of patch side in pixels.
        - level2tile: int, level to tile.

    Returns:
        - coordinates: list of tuples, (x, y, level, size) of patches at
        level2tile.

    """
    imlowres = numpy.array(slide.read_region((0, 0), level2tile, slide.level_dimensions[level2tile]))[:, :, 0:3]
    masklowres = get_tissue(imlowres)

    posi, posj = naive_patch_placement_optimization(masklowres, patchsize, verbose=verbose)

    posi *= (2 ** level2tile)
    posj *= (2 ** level2tile)

    coordinates = []

    for i, j in zip(posi, posj):

        coordinates.append((j, i, level2tile, patchsize))

    return coordinates


class PatchTree:
    """
    Convenient structure to store hierarchical patch representation of a slide.

    ***************************************************************************
    """

    def __init__(self, slide, patchsize, levelmax, levelmin, verbose=False):
        """
        Instantiate object, build trees.

        ********************************
        """
        self.slide = slide
        self.patchsize = patchsize
        self.levelmax = levelmax
        self.levelmin = levelmin
        self.deltaread = int(numpy.log2(1024 / patchsize))

        self.inipatches = tile_at_level(self.slide, self.patchsize, self.levelmax, verbose=verbose)

        print('number of initial patches: ', len(self.inipatches))

        self.parents = dict()
        self.children = dict()
        self.children_read = dict()
        self.predictions = dict()
        self.variances = dict()
        self.warnings = dict()

        for patch in self.inipatches:

            self.parents[patch] = None

        self.build()

    def build_patch_tree(self, nodes):
        """
        Build patch tree.

        *****************
        """
        level = nodes[0][2]

        if level > self.levelmin:

            next_nodes = []

            for node in nodes:

                children = get_patch_children(node)

                next_nodes += children

                self.children[node] = children

                for child in children:

                    self.parents[child] = node

            self.build_patch_tree(next_nodes)

    def set_children2read(self, node, level2read):
        """
        Build children for read tree.

        *****************************
        """
        if node not in self.children_read.keys():

            self.children_read[node] = dict()

        read_level_dico = dict()

        xstart, ystart, level, size = node

        deltaread = level - level2read
        m_shape = (self.patchsize * (2 ** deltaread), self.patchsize * (2 ** deltaread))

        read_level_dico['imsize'] = m_shape

        imy, imx = gen_patch_coords(m_shape, self.patchsize, 0, 0)
        posy = imy * (2 ** level2read)
        posx = imx * (2 ** level2read)
        posx += xstart
        posy += ystart

        read_level_dico['imcoords'] = dict()

        for k in range(len(posy)):
            x = posx[k]
            y = posy[k]
            xim = imx[k]
            yim = imy[k]
            key = (x, y, level2read, self.patchsize)

            if key in self.children or key in self.parents:
                read_level_dico['imcoords'][(xim, yim)] = key

        self.children_read[node][level2read] = read_level_dico

    def build_read_tree(self, nodes):
        """
        Build single-node sub-tree.

        ***************************
        """
        level = nodes[0][2]
        level2read = level - self.deltaread

        if level2read >= self.levelmin:

            children = []

            if level == self.levelmax:

                for l in range(level2read, self.levelmax + 1):
                    for node in nodes:
                        self.set_children2read(node, l)
                        children += self.children[node]

            # other cases, do the job exactly one time
            else:

                for node in nodes:
                    self.set_children2read(node, level2read)
                    children += self.children[node]

            self.build_read_tree(children)

    def build(self):
        """
        Build the patch tree with recursive procedures.

        ***********************************************
        """
        self.build_patch_tree(self.inipatches)
        self.build_read_tree(self.inipatches)

    def images_in_node_at_level(self, node, level, warnings):
        """
        Yield images in a node at a given level.

        ****************************************
        """
        if node not in self.children_read.keys():

            return []

        patchpyramid = self.children_read[node]

        if level not in patchpyramid:

            return []

        rootx, rooty, rootlevel, rootsize = node

        patchpyramidlevel = patchpyramid[level]

        imagesize = patchpyramidlevel['imsize']

        image = self.slide.read_region((rootx, rooty), level, imagesize)
        image = numpy.array(image)[:, :, 0:3]

        for coord in patchpyramidlevel['imcoords']:

            x, y = coord

            im = image[y:y + rootsize, x:x + rootsize]

            if warnings:
                mask = get_tissue(im)
                self.warnings[patchpyramidlevel['imcoords'][coord]] = (mask.sum() < 0.7 * rootsize * rootsize)

            yield coord, patchpyramidlevel['imcoords'][coord], im

    def images_at_level(self, level, warnings=True):
        """
        Yield images at a given level.

        ******************************
        """
        readlevel = level + self.deltaread

        # if level + deltaread >= levelmax, use inipatches as nodes
        if readlevel >= self.levelmax:

            nodes = self.inipatches

        # if level + deltaread < levelmax, we have to get all nodes at level + deltaread
        else:

            nodes = [node for node in self.parents if node[2] == readlevel]

        for node in nodes:

            # get a batch of patches, one batch is a floor in the pyramid of root node
            patchnimlist = [patchnim for patchnim in self.images_in_node_at_level(node, level, warnings=warnings)]
            relpatchlist = [patchnim[0] for patchnim in patchnimlist]
            abspatchlist = [patchnim[1] for patchnim in patchnimlist]
            imlist = [patchnim[2] for patchnim in patchnimlist]

            # yield batches of patches
            yield relpatchlist, abspatchlist, imlist
