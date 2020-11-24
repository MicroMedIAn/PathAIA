# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""
import numpy
import openslide
from ..util.paths import slides_in_folder, slide_basename
from ..util.images import regular_grid
from ..utils.basic import ifnone
from .visu import preview_from_queries
from .filters import (
    filter_hasdapi,
    filter_has_significant_dapi,
    filter_has_tissue_he,
    standardize_filters,
)
import os
import csv
from skimage.io import imsave
import shutil
import warnings
from tqdm import tqdm
from .errors import UnknownFilterError


izi_filters = {
    "has-dapi": filter_hasdapi,
    "has-significant-dapi": filter_has_significant_dapi,
    "has-tissue-he": filter_has_tissue_he,
}


def filter_image(image, filters):
    """
    Check multiple filters on an image.

    Args:
        image (ndarray): the patch to be filtered.
        filters (list of function): functions that turn images into booleans.

    Returns:
        acceptable: bool, whether an image has passed every filters.

    """
    for filt in filters:
        if callable(filt):
            if not filt(image):
                return False
        elif type(filt) == str:
            if not izi_filters[filt](image):
                return False
        else:
            raise UnknownFilterError("{} is not a valid filter !!!".format(filt))
    return True


def slide_rois(
    slide, level, psize, interval, ancestors=None, offset=None, filters=None
):
    """
    Return the absolute coordinates of patches.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels.

    Args:
        slide (OpenSlide): the slide to patchify.
        level (int): pyramid level.
        psize (int): size of the side of the patch (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        ancestors (list of patch dict): patches that contain upcoming patches.
        offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
        filters (list of func): filters to accept patches.

    Yields:
        ndarray: numpy array rgb image.
        tuple of ndarray: icoords, jcoords.

    """
    ancestors = ifnone(ancestors, [])
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, [])
    if len(ancestors) > 0:
        mag = slide.level_downsamples[level]
        shape = dict()
        shape["x"] = int(ancestors[0]["dx"] / mag)
        shape["y"] = int(ancestors[0]["dy"] / mag)
        dx = int(psize * mag)
        dy = int(psize * mag)
        patches = []
        for ancestor in ancestors:
            # ancestor is a patch
            rx, ry = ancestor["x"], ancestor["y"]
            prefix = ancestor["id"]
            k = 0
            for patch in regular_grid(shape, interval):
                k += 1
                idx = "{}#{}".format(prefix, k)
                y = int(patch["y"] * mag + ry)
                x = int(patch["x"] * mag + rx)
                patches.append(
                    {
                        "id": idx,
                        "x": x,
                        "y": y,
                        "level": level,
                        "dx": dx,
                        "dy": dy,
                        "parent": prefix,
                    }
                )
        for patch in tqdm(patches):
            try:
                image = slide.read_region(
                    (patch["x"], patch["y"]), patch["level"], (psize, psize)
                )
                image = numpy.array(image)[:, :, 0:3]
                if filter_image(image, filters):
                    yield patch, image
            except openslide.lowlevel.OpenSlideError:
                print(
                    "small failure while reading tile x={}, y={} in {}".format(
                        patch["x"], patch["y"], slide._filename
                    )
                )
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
                    yield {
                        "id": idx,
                        "x": x,
                        "y": y,
                        "level": level,
                        "dx": dx,
                        "dy": dy,
                        "parent": "None",
                    }, image
            except openslide.lowlevel.OpenSlideError:
                print(
                    "small failure while reading tile x={}, y={} in {}".format(
                        x, y, slide._filename
                    )
                )


def patchify_slide(
    slidefile, outdir, level, psize, interval, offset=None, filters=None, verbose=2
):
    """
    Save patches of a given wsi.

    Args:
        slidefile (str): abs path to slide file.
        outdir (str): abs path to an output folder.
        level (int): pyramid level.
        psize (int): size of the side of the patches (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
        filters (list of func): filters to accept patches.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, [])
    # Get name of the slide
    slide_id = slide_basename(slidefile)
    # if output directory has the same name, it's ok
    if os.path.basename(outdir) == slide_id:
        slide_folder_output = outdir
    # otherwise, create a subfolder with the name of the slide
    else:
        slide_folder_output = os.path.join(outdir, slide_id)
        if os.path.isdir(slide_folder_output):
            shutil.rmtree(slide_folder_output, ignore_errors=True)
        os.makedirs(slide_folder_output)

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
    outleveldir = os.path.join(slide_folder_output, "level_{}".format(level))
    if os.path.isdir(outleveldir):
        shutil.rmtree(outleveldir, ignore_errors=True)
    os.makedirs(outleveldir)
    ########################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for data, img in slide_rois(
            slide, level, psize, interval, offset=offset, filters=filters
        ):
            outfile = os.path.join(
                outleveldir, "{}_{}_{}.png".format(data["x"], data["y"], data["level"])
            )
            imsave(outfile, img)
            plist.append(data)
    if verbose > 1:
        print("end of patchification.")
        print("starting metadata csv export...")
    csv_columns = ["id", "parent", "level", "x", "y", "dx", "dy"]
    csv_path = os.path.join(slide_folder_output, "patches.csv")
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        writer.writerows(plist)
    if verbose > 1:
        print("end of metadata export.")
        print("starting thumbnail export...")
        out_thumbnailfile = os.path.join(outleveldir, "thumbnail.png")
        thumbnail = preview_from_queries(slide, plist)
        imsave(out_thumbnailfile, thumbnail)
        print("ending thumbnail export.")


def patchify_slide_hierarchically(
    slidefile,
    outdir,
    top_level,
    low_level,
    psize,
    interval,
    offset=None,
    filters=None,
    silent=None,
    verbose=2,
):
    """
    Save patches of a given wsi in a hierarchical way.

    Args:
        slidefile (str): abs path to a slide file.
        outdir (str): abs path to an output folder.
        top_level (int): top pyramid level to consider.
        low_level (int): lowest pyramid level to consider.
        psize (int): size of the side of the patches (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
        filters (dict of list of func): filters to accept patches.
        silent (list of int): pyramid level not to output.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, {})
    silent = ifnone(silent, [])
    level_filters = standardize_filters(filters, top_level, low_level)
    # Get name of the slide
    slide_id = slide_basename(slidefile)
    # if output directory has the same name, it's ok
    if os.path.basename(outdir) == slide_id:
        slide_folder_output = outdir
    # otherwise, create a subfolder with the name of the slide
    else:
        slide_folder_output = os.path.join(outdir, slide_id)
        if os.path.isdir(slide_folder_output):
            shutil.rmtree(slide_folder_output, ignore_errors=True)
        os.makedirs(slide_folder_output)

    csv_columns = ["id", "parent", "level", "x", "y", "dx", "dy"]
    csv_path = os.path.join(slide_folder_output, "patches.csv")
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
                    print("filtering: {}".format(level_filters[level]))
                    print("ancestors: {} patches".format(len(plist)))
                    print("starting patchification...")
            current_plist = []
            # level directory
            outleveldir = os.path.join(slide_folder_output, "level_{}".format(level))
            if os.path.isdir(outleveldir):
                shutil.rmtree(outleveldir, ignore_errors=True)
            os.makedirs(outleveldir)
            ########################
            if level not in silent:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for data, img in slide_rois(
                        slide,
                        level,
                        psize,
                        interval,
                        ancestors=plist,
                        offset=offset,
                        filters=level_filters[level],
                    ):
                        outfile = os.path.join(
                            outleveldir,
                            "{}_{}_{}.png".format(data["x"], data["y"], data["level"]),
                        )
                        imsave(outfile, img)
                        current_plist.append(data)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for data, img in slide_rois(
                        slide,
                        level,
                        psize,
                        interval,
                        ancestors=plist,
                        offset=offset,
                        filters=level_filters[level],
                    ):
                        # outfile = os.path.join(outleveldir, "{}_{}_{}.png".format(data["x"], data["y"], data["level"]))
                        # imsave(outfile, img)
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


def patchify_folder(
    infolder, outfolder, level, psize, interval, offset=None, filters=None, verbose=2
):
    """
    Save patches of all wsi inside a folder.

    Args:
        infolder (str): abs path to a folder of slides.
        outfolder (str): abs path to an output folder.
        level (int): pyramid level.
        psize (int): size of the side of the patches (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
        filters (list of func): filters to accept patches.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, [])
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
        patchify_slide(
            slidefile,
            outdir,
            level,
            psize,
            interval,
            offset=offset,
            filters=filters,
            verbose=verbose,
        )


def patchify_folder_hierarchically(
    infolder,
    outfolder,
    top_level,
    low_level,
    psize,
    interval,
    offset=None,
    filters=None,
    silent=None,
    verbose=2,
):
    """
    Save hierarchical patches of all wsi inside a folder.

    Args:
        infolder (str): abs path to a folder of slides.
        outfolder (str): abs path to an output folder.
        top_level (int): top pyramid level to consider.
        low_level (int): lowest pyramid level to consider.
        psize (int): size of the side of the patches (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
        filters (dict of list of func): filters to accept patches.
        silent (list of int): pyramid level not to output.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, {})
    silent = ifnone(silent, [])
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
        patchify_slide_hierarchically(
            slidefile,
            outdir,
            top_level,
            low_level,
            psize,
            interval,
            offset=offset,
            filters=filters,
            silent=silent,
            verbose=verbose,
        )
