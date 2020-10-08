# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""
import numpy
import openslide
from skimage.color import rgb2lab
from .util import regular_grid, slides_in_folder, slide_basename
from .visu import preview_from_queries
from .filters import filter_hasdapi, filter_has_significant_dapi
import os
import csv
from skimage.io import imsave
import shutil


izi_filters = {"has-dapi": filter_hasdapi,
               "has-significant-dapi": filter_has_significant_dapi}


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


def filter_image(image, filters):
    """
    Check multiple filters on an image.

    Arguments:
        - image: numpy array, patch to filter.
        - filters: list of functions, image to bool.

    Returns:
        - acceptable: bool, whether an image has passed every filters.

    """
    for filt in filters:
        if callable(filt):
            if not filt(image):
                return False
        elif type(filt) == str:
            if not izi_filters[filt](image):
                return False
        else:
            raise UnknownMethodError("{} is not a valid filter !!!".format(filt))
    return True


def slide_rois(slide, level, psize, interval, ancestors=[], offset={"x": 0, "y": 0}, filters=[]):
    """
    Return the absolute coordinates of patches.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels.

    Arguments:
        - slide: openslide object.
        - level: int, pyramid level.
        - psize: int
        - interval: dictionary, {"x", "y"} interval between 2 neighboring patches.
        - offset: dictionary, {"x", "y"} offset in px on x and y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.

    Yields:
        - image: numpy array rgb image.
        - coords: tuple of numpy arrays, (icoords, jcoords).

    """
    if len(ancestors) > 0:
        for ancestor in ancestors:
            # ancestor is a patch
            rx, ry = ancestor["x"], ancestor["y"]
            mag = slide.level_downsamples[level]
            shape = dict()
            shape["x"] = int(ancestor["dx"] / mag)
            shape["y"] = int(ancestor["dy"] / mag)
            prefix = ancestor["id"]
            k = 0
            for patch in regular_grid(shape, interval):
                k += 1
                idx = "{}#{}".format(prefix, k)
                y = int(patch["y"] * mag + rx)
                x = int(patch["x"] * mag + ry)
                dx = int(psize * mag)
                dy = int(psize * mag)
                try:
                    image = slide.read_region((x, y), level, (psize, psize))
                    image = numpy.array(image)[:, :, 0:3]
                    if filter_image(image, filters):
                        yield {"id": idx,
                               "x": x,
                               "y": y,
                               "level": level,
                               "dx": dx,
                               "dy": dy,
                               "parent": prefix}, image
                except openslide.lowlevel.OpenSlideError:
                    print("small failure while reading tile x={}, y={} in {}".format(x, y, slide._filename))
    else:
        shape = dict()
        shape["x"], shape["y"] = slide.level_dimensions[level]
        mag = slide.level_downsamples[level]
        k = 0
        for patch in regular_grid(shape, interval):
            k += 1
            idx = "#{}".format(k)
            y = int(patch["y"] * mag + offset["y"])
            x = int(patch["x"] * mag + offset["x"])
            dx = int(psize * mag)
            dy = int(psize * mag)
            try:
                image = slide.read_region((x, y), level, (psize, psize))
                image = numpy.array(image)[:, :, 0:3]
                if filter_image(image, filters):
                    yield {"id": idx,
                           "x": x,
                           "y": y,
                           "level": level,
                           "dx": dx,
                           "dy": dy,
                           "parent": "None"}, image
            except openslide.lowlevel.OpenSlideError:
                print("small failure while reading tile x={}, y={} in {}".format(x, y, slide._filename))


def patchify_slide(slidefile,
                   outdir,
                   level,
                   psize,
                   interval,
                   offset={"x": 0, "y": 0},
                   filters=[],
                   verbose=2):
    """
    Save patches of a given wsi.

    Arguments:
        - slidefile: str, abs path to slide file.
        - outdir: str, abs path to an output folder.
        - level: int, pyramid level.
        - psize: int
        - interval: dictionary, {"x", "y"} interval between 2 neighboring patches.
        - offset: dictionary, {"x", "y"} offset in px on x and y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.
        - tissue: bool, whether to filter on tissue.
        - verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    if verbose > 0:
        print("patchifying: {}".format(slidefile))
        if verbose > 1:
            print("level: {}".format(level))
            print("patch-size: {}".format(psize))
            print("interval: {}".format(interval))
            print("offset: {}".format(offset))
            print("filtering: {}".format(filters))
            print("starting patchification...")
    slide = openslide.OpenSlide(slidefile)
    plist = []
    # level directory
    outleveldir = os.path.join(outdir, "level_{}".format(level))
    if os.path.isdir(outleveldir):
        shutil.rmtree(outleveldir, ignore_errors=True)
    os.makedirs(outleveldir)
    ########################
    for data, img in slide_rois(slide, level, psize, interval, offset=offset, filters=filters):
        outfile = os.path.join(outleveldir, "{}_{}_{}.png".format(data["x"], data["y"], data["level"]))
        imsave(outfile, img)
        plist.append(data)
    if verbose > 1:
        print("end of patchification.")
        print("starting metadata csv export...")
    csv_columns = ["id", "parent", "level", "x", "y", "dx", "dy"]
    csv_path = os.path.join(outdir, "patches.csv")
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        writer.writerows(plist)
    if verbose > 1:
        print("end of metadata export.")
        print("starting thumbnail export...")
        out_thumbnailfile = os.path.join(outdir, "thumbnail.png")
        thumbnail = preview_from_queries(slide, plist)
        imsave(out_thumbnailfile, thumbnail)
        print("ending thumbnail export.")


def patchify_slide_hierarchically(slidefile,
                                  outdir,
                                  top_level,
                                  low_level,
                                  psize,
                                  interval,
                                  offset={"x": 0, "y": 0},
                                  filters=[],
                                  verbose=2):
    """
    Save patches of a given wsi.

    Arguments:
        - slidefile: str, abs path to slide file.
        - outdir: str, abs path to an output folder.
        - level: int, pyramid level.
        - psize: int
        - interval: dictionary, {"x", "y"} interval between 2 neighboring patches.
        - offset: dictionary, {"x", "y"} offset in px on x and y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.
        - tissue: bool, whether to filter on tissue.
        - verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    csv_columns = ["id", "parent", "level", "x", "y", "dx", "dy"]
    csv_path = os.path.join(outdir, "patches.csv")
    slide = openslide.OpenSlide(slidefile)
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        plist = []
        current_plist = []
        for level in range(top_level, low_level - 1, -1):
            if verbose > 0:
                print("patchifying: {}".format(slidefile))
                if verbose > 1:
                    print("level: {}".format(level))
                    print("patch-size: {}".format(psize))
                    print("interval: {}".format(interval))
                    print("offset: {}".format(offset))
                    print("filtering: {}".format(filters))
                    print("ancestors: {} patches".format(len(plist)))
                    print("starting patchification...")
            current_plist = []
            # level directory
            outleveldir = os.path.join(outdir, "level_{}".format(level))
            if os.path.isdir(outleveldir):
                shutil.rmtree(outleveldir, ignore_errors=True)
            os.makedirs(outleveldir)
            ########################
            for data, img in slide_rois(slide, level, psize, interval, ancestors=plist, offset=offset, filters=filters):
                outfile = os.path.join(outleveldir, "{}_{}_{}.png".format(data["x"], data["y"], data["level"]))
                imsave(outfile, img)
                current_plist.append(data)
            plist = [p for p in current_plist]
            if verbose > 1:
                print("end of patchification.")
                print("starting metadata csv export...")
            writer.writerows(plist)
            if verbose > 1:
                print("end of metadata export.")
                print("starting thumbnail export...")
                out_thumbnailfile = os.path.join(outleveldir, "thumbnail.png")
                thumbnail = preview_from_queries(slide, current_plist)
                imsave(out_thumbnailfile, thumbnail)
                print("ending thumbnail export.")


def patchify_folder(infolder, outfolder, level, psize, interval, offset={"x": 0, "y": 0}, filters=[], verbose=2):
    """
    Save patches of all wsi inside a folder.

    Arguments:
        - slidefile: str, abs path to slide file.
        - outdir: str, abs path to an output folder.
        - level: int, pyramid level.
        - psize: int
        - interval: dictionary, {"x", "y"} interval between 2 neighboring patches.
        - offset: dictionary, {"x", "y"} offset in px on x and y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.
        - tissue: bool, whether to filter on tissue.
        - verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    slidefiles = slides_in_folder(infolder)
    total = len(slidefiles)
    k = 0
    for slidefile in slidefiles:
        k += 1
        if verbose > 0:
            print("slide {} / {}".format(k, total))
        slidename = slide_basename(slidefile)
        outdir = os.path.join(outfolder, slidename)
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)
        patchify_slide(slidefile, outdir, level, psize, interval, offset=offset, filters=filters, verbose=verbose)


def patchify_folder_hierarchically(infolder, outfolder, top_level, low_level, psize, interval, offset={"x": 0, "y": 0}, filters=[], verbose=2):
    """
    Save patches of all wsi inside a folder.

    Arguments:
        - slidefile: str, abs path to slide file.
        - outdir: str, abs path to an output folder.
        - level: int, pyramid level.
        - psize: int
        - interval: dictionary, {"x", "y"} interval between 2 neighboring patches.
        - offset: dictionary, {"x", "y"} offset in px on x and y axis for patch start.
        - coords: bool, coordinates of patches will be yielded if set to True.
        - tissue: bool, whether to filter on tissue.
        - verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    slidefiles = slides_in_folder(infolder)
    total = len(slidefiles)
    k = 0
    for slidefile in slidefiles:
        k += 1
        if verbose > 0:
            print("slide {} / {}".format(k, total))
        slidename = slide_basename(slidefile)
        outdir = os.path.join(outfolder, slidename)
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)
        patchify_slide_hierarchically(slidefile, outdir, top_level, low_level, psize, interval, offset=offset, filters=filters, verbose=verbose)
