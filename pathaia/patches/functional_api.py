# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""
import warnings
import numpy
import openslide
from ..util.paths import slides_in_folder, slide_basename, safe_rmtree, get_files
from ..util.images import regular_grid
from ..util.basic import ifnone
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
from skimage.filters import threshold_otsu
from tqdm import tqdm
from .errors import UnknownFilterError
from pathlib import Path


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
        bool: whether an image has passed every filters.

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
        shape["x"] = round(ancestors[0]["dx"] / mag)
        shape["y"] = round(ancestors[0]["dy"] / mag)
        dx = int(psize * mag)
        dy = int(psize * mag)
        patches = []
        for ancestor in ancestors:
            # ancestor is a patch
            rx, ry = ancestor["x"], ancestor["y"]
            prefix = ancestor["id"]
            k = 0
            for patch in regular_grid(shape, interval, psize):
                k += 1
                idx = "{}#{}".format(prefix, k)
                y = round(patch["y"] * mag + ry)
                x = round(patch["x"] * mag + rx)
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
        for patch in tqdm(patches, ascii=True):
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
        for patch in regular_grid(shape, interval, psize):
            k += 1
            idx = "#{}".format(k)
            y = round(patch["y"] * mag + offset["y"])
            x = round(patch["x"] * mag + offset["x"])
            dx = round(psize * mag)
            dy = round(psize * mag)
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
    slidefile,
    outdir,
    level,
    psize,
    interval,
    offset=None,
    filters=None,
    erase_tree=None,
    verbose=2,
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
        erase_tree (bool): whether to erase outfolder if it exists. If None, user will be prompted for a choice.
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
            erase_tree = safe_rmtree(
                slide_folder_output, ignore_errors=True, erase_tree=erase_tree
            )
        os.makedirs(slide_folder_output, exist_ok=True)

    if verbose > 0:
        print("patchifying: {}".format(slidefile))
        if verbose > 1:
            print("level: {}".format(level))
            print("patch-size: {}".format(psize))
            print("interval: {}".format(interval))
            print("offset: {}".format(offset))
            print("filtering: {}".format(filters))
            print("starting patchification...")
    slide = openslide.OpenSlide(str(slidefile))
    plist = []
    # level directory
    outleveldir = os.path.join(slide_folder_output, "level_{}".format(level))
    if os.path.isdir(outleveldir):
        safe_rmtree(outleveldir, ignore_errors=True, erase_tree=erase_tree)
    os.makedirs(outleveldir, exist_ok=True)
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
    erase_tree=None,
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
        erase_tree (bool): whether to erase outfolder if it exists. If None, user will be prompted for a choice.
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
            erase_tree = safe_rmtree(
                slide_folder_output, ignore_errors=True, erase_tree=erase_tree
            )
        os.makedirs(slide_folder_output, exist_ok=True)

    csv_columns = ["id", "parent", "level", "x", "y", "dx", "dy"]
    csv_path = os.path.join(slide_folder_output, "patches.csv")
    slide = openslide.OpenSlide(str(slidefile))
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
                safe_rmtree(outleveldir, ignore_errors=True, erase_tree=erase_tree)
            os.makedirs(outleveldir, exist_ok=True)
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
    infolder,
    outfolder,
    level,
    psize,
    interval,
    offset=None,
    filters=None,
    extensions=(".mrxs",),
    recurse=False,
    folders=None,
    erase_tree=None,
    verbose=2,
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
        extensions (list of str): list of file extensions to consider. Defaults to '.mrxs'.
        recurse (bool): whether to look for files recursively.
        folders (list of str): list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree (bool): whether to erase outfolder if it exists. If None, user will be prompted for a choice.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    if os.path.isdir(outfolder):
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, [])
    slidefiles = get_files(
        infolder, extensions=extensions, recurse=recurse, folders=folders
    ).map(str)
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
            safe_rmtree(outdir, ignore_errors=True, erase_tree=erase_tree)
        os.makedirs(outdir, exist_ok=True)
        # patchify folder must be robust to 'missing image data' rare cases...
        try:
            patchify_slide(
                slidefile,
                outdir,
                level,
                psize,
                interval,
                offset=offset,
                filters=filters,
                erase_tree=erase_tree,
                verbose=verbose,
            )
        except (openslide.OpenSlideUnsupportedFormatError,
                openslide.lowlevel.OpenSlideError) as e:
            warnings.warn(str(e))


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
    extensions=(".mrxs",),
    recurse=False,
    folders=None,
    erase_tree=None,
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
        extensions (list of str): list of file extensions to consider. Defaults to '.mrxs'.
        recurse (bool): whether to look for files recursively.
        folders (list of str): list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree (bool): whether to erase outfolder if it exists. If None, user will be prompted for a choice.
        verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """
    if os.path.isdir(outfolder):
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, {})
    silent = ifnone(silent, [])
    slidefiles = get_files(
        infolder, extensions=extensions, recurse=recurse, folders=folders
    ).map(str)
    total = len(slidefiles)
    k = 0
    for slidefile in slidefiles:
        k += 1
        if verbose > 0:
            print("slide {} / {}".format(k, total))
        slidename = slide_basename(slidefile)
        outdir = os.path.join(outfolder, slidename)
        if os.path.isdir(outdir):
            safe_rmtree(outdir, ignore_errors=True, erase_tree=erase_tree)
        os.makedirs(outdir, exist_ok=True)
        try:
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
                erase_tree=erase_tree,
                verbose=verbose,
            )
        except (openslide.OpenSlideUnsupportedFormatError,
                openslide.lowlevel.OpenSlideError) as e:
            warnings.warn(str(e))


def extract_tissue_patch_coords(
    infolder,
    outfolder,
    level,
    psize,
    interval,
    extensions=(".mrxs",),
    recurse=True,
    folders=None,
    erase_tree=None,
):
    """
    Extracts all patch coordinates that contain tissue at aspecific level from WSI files
    in a folder and stores them in csvs. Foreground is evaluated using otsu thresholding.

    Args:
        infolder (str): abs path to a folder of slides.
        outfolder (str): abs path to a folder to stroe output csv files.
        level (int): pyramid level to consider.
        psize (int): size of the side of the patches (in pixels).
        interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
        extensions (list of str): list of file extensions to consider. Defaults to '.mrxs'.
        recurse (bool): whether to look for files recursively.
        folders (list of str): list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree (bool): whether to erase outfolder if it exists. If None, user will be prompted for a choice.
    """
    outfolder = Path(outfolder)
    if outfolder.is_dir():
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
    outfolder.mkdir(parents=True, exist_ok=True)
    overlap_size = psize - interval
    files = get_files(infolder, extensions=extensions, recurse=recurse, folders=folders)

    for file in files:
        print(file.stem)
        slide = openslide.OpenSlide(str(file))
        dsr = int(slide.level_downsamples[level])
        psize_0 = dsr * psize
        w, h = slide.dimensions

        thumb_w = int((w / dsr - overlap_size) / interval)
        thumb_h = int((h / dsr - overlap_size) / interval)
        thumb = slide.get_thumbnail((thumb_w, thumb_h))
        thumb = numpy.array(thumb.convert("L"))
        thr = threshold_otsu(thumb[thumb > 0])

        outfile = outfolder / (file.stem + ".csv")
        with open(outfile, "w") as f:
            writer = csv.DictWriter(f, ["id", "parent", "level", "x", "y", "dx", "dy"])
            writer.writeheader()
            for k, (y, x) in enumerate(numpy.argwhere((thumb > 0) & (thumb < thr))):
                x = x * interval * dsr
                y = y * interval * dsr
                writer.writerow(
                    {
                        "id": f"#{k}",
                        "level": level,
                        "x": x,
                        "y": y,
                        "dx": psize_0,
                        "dy": psize_0,
                        "parent": "None",
                    }
                )
