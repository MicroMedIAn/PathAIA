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
from ..util.images import regular_grid, get_coords_from_mask
from ..util.basic import ifnone
from ..util.types import Filter, FilterList, PathLike, Patch, NDImage, NDBoolMask
from .visu import preview_from_queries
from .filters import (
    filter_hasdapi,
    filter_has_significant_dapi,
    filter_has_tissue_he,
    standardize_filters,
)
from .slide_filters import filter_thumbnail
import os
import csv
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
from .errors import UnknownFilterError
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Iterator


izi_filters = {
    "has-dapi": filter_hasdapi,
    "has-significant-dapi": filter_has_significant_dapi,
    "has-tissue-he": filter_has_tissue_he,
}

slide_filters = {"full": filter_thumbnail}


def filter_image(image: NDImage, filters: Sequence[Filter]) -> bool:
    """
    Check multiple filters on an image.

    Args:
        image: the patch to be filtered.
        filters: functions that turn images into booleans.

    Returns:
        Whether an image has passed every filters.

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


def apply_slide_filters(thumb: NDImage, filters: Sequence[Filter]) -> NDBoolMask:
    """
    Apply all filters to input thumbnail. Performs logical and between output masks.

    Args:
        thumb: thumbnail to compute mask from.
        filters: list of filters to apply to thumb. Each
            filter should output a boolean mask with same dimensions as thumb.

    Returns:
        Boolean mask where tissue pixels are True.
    """
    mask = numpy.ones(thumb.shape[:2], dtype=bool)
    for filt in filters:
        if isinstance(filt, str):
            filt = slide_filters[filt]
        mask = mask & filt(thumb)
    return mask


def slide_rois(
    slide: openslide.OpenSlide,
    level: int,
    psize: int,
    interval: Dict[str, int],
    ancestors: Optional[Sequence[Patch]] = None,
    offset: Optional[Sequence[Dict[str, int]]] = None,
    filters: Optional[Sequence[Filter]] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
) -> Iterator[Tuple[Patch, NDImage]]:
    """
    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels, get patches with its coordinates.

    Args:
        slide: the slide to patchify.
        level: pyramid level.
        psize: size of the side of the patch (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        ancestors: patches that contain upcoming patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean mask.

    Yields:
        A tuple containing a dict describing a patch and the corresponding image as
        ndarray. The dict contains the patch's id, ancestor's id if relevant,
        coordinates (at level 0), level and size (at level 0).

    Example::
        >>> level = 1
        >>> psize = 224
        >>> interval = {"x": 224, "y": 224}
        >>> next(slide_rois(slide, level, psize, interval))
        "id": "#1",
        "x": 0,
        "y": 0,
        "level": 1,
        "dx": 448,
        "dy": 448,
        "parent": "None"

    """
    ancestors = ifnone(ancestors, [])
    offset = ifnone(offset, {"x": 0, "y": 0})
    filters = ifnone(filters, [])
    slide_filters = ifnone(slide_filters, [])
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
        thumb = numpy.array(slide.get_thumbnail((thumb_size, thumb_size)))
        mask = apply_slide_filters(thumb, slide_filters)
        k = 0
        for patch in get_coords_from_mask(mask, shape, interval, psize):
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
    slidefile: PathLike,
    outdir: PathLike,
    level: int,
    psize: int,
    interval: Dict[str, int],
    offset: Optional[Dict[str, int]] = None,
    filters: Optional[Sequence[Filter]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
):
    """
    Save patches of a given wsi.

    Args:
        slidefile: abs path to slide file.
        outdir: abs path to an output folder.
        level: pyramid level.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean
            mask.
        verbose (int: 0 => nada, 1 => patchifying parameters, 2 => start-end of
            processes, thumbnail export.

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
            slide,
            level,
            psize,
            interval,
            offset=offset,
            filters=filters,
            thumb_size=thumb_size,
            slide_filters=slide_filters,
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
    slidefile: PathLike,
    outdir: PathLike,
    top_level: int,
    low_level: int,
    psize: int,
    interval: Dict[str, int],
    offset: Optional[Dict[str, int]] = None,
    filters: Optional[FilterList] = None,
    silent: Optional[Sequence[int]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
):
    """
    Save patches of a given wsi in a hierarchical way.

    Args:
        slidefile: abs path to a slide file.
        outdir: abs path to an output folder.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        silent: pyramid level not to output.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean
            mask.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.

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
                    thumb_size=thumb_size,
                    slide_filters=slide_filters,
                ):
                    if level not in silent:
                        outfile = os.path.join(
                            outleveldir,
                            "{}_{}_{}.png".format(data["x"], data["y"], data["level"]),
                        )
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


def patchify_folder(
    infolder: str,
    outfolder: str,
    level: int,
    psize: int,
    interval: Dict[str, int],
    offset: Optional[Dict[str, int]] = None,
    filters: Optional[Sequence[Filter]] = None,
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = False,
    folders: Optional[Sequence[str]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
):
    """
    Save patches of all wsi inside a folder.

    Args:
        infolder: abs path to a folder of slides.
        outfolder: abs path to an output folder.
        level: pyramid level.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        recurse: whether to look for files recursively.
        folders: list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean
            mask.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.

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
                thumb_size=thumb_size,
                slide_filters=slide_filters,
                verbose=verbose,
            )
        except (
            openslide.OpenSlideUnsupportedFormatError,
            openslide.lowlevel.OpenSlideError,
        ) as e:
            warnings.warn(str(e))


def patchify_folder_hierarchically(
    infolder: PathLike,
    outfolder: PathLike,
    top_level: int,
    low_level: int,
    psize: int,
    interval: Dict[str, int],
    offset: Optional[Dict[str, int]] = None,
    filters: Optional[FilterList] = None,
    silent: Optional[Sequence[int]] = None,
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = False,
    folders: Optional[Sequence[str]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
):
    """
    Save hierarchical patches of all wsi inside a folder.

    Args:
        infolder: abs path to a folder of slides.
        outfolder: abs path to an output folder.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        silent: pyramid level not to output.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        recurse: whether to look for files recursively.
        folders: list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.

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
                thumb_size=thumb_size,
                slide_filters=slide_filters,
                verbose=verbose,
            )
        except (
            openslide.OpenSlideUnsupportedFormatError,
            openslide.lowlevel.OpenSlideError,
        ) as e:
            warnings.warn(str(e))


def extract_tissue_patch_coords(
    infolder: PathLike,
    outfolder: PathLike,
    level: int,
    psize: int,
    interval: Dict[str, int],
    thumb_size: int = 512,
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = True,
    folders: Optional[Sequence[str]] = None,
    erase_tree: Optional[bool] = None,
    filters: Optional[Sequence[Filter]] = None,
    save_thumbs: bool = False,
):
    """
    Extracts all patch coordinates that contain tissue at aspecific level from WSI files
    in a folder and stores them in csvs. Foreground is evaluated using otsu thresholding.

    Args:
        infolder: abs path to a folder of slides.
        outfolder: abs path to a folder to stroe output csv files.
        level: pyramid level to consider.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        recurse: whether to look for files recursively.
        folders: list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree: whether to erase outfolder if it exists. If None, user will be prompted for a choice.
        filters: list of filters to apply to thumbnail. Should output boolean mask.
        save_thumbs: save masked thumbnails of extracted zones.
    """
    outfolder = Path(outfolder)
    if outfolder.is_dir():
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
    outfolder.mkdir(parents=True, exist_ok=True)
    if save_thumbs:
        thumbfolder = outfolder / "thumbnails"
        thumbfolder.mkdir()
    overlap_size = psize - interval
    files = get_files(infolder, extensions=extensions, recurse=recurse, folders=folders)
    filters = ifnone(filters, [])

    for file in files:
        print(file.stem)
        slide = openslide.OpenSlide(str(file))
        dsr = int(slide.level_downsamples[level])
        psize_0 = dsr * psize
        w, h = slide.dimensions

        thumb = numpy.array(slide.get_thumbnail((thumb_size, thumb_size)))
        mask = apply_slide_filters(thumb, filters)
        if save_thumbs:
            imsave(thumbfolder / f"{file.stem}.png", mask[..., None] * thumb)
        mask_w = int((w / dsr - overlap_size) / interval)
        mask_h = int((h / dsr - overlap_size) / interval)
        mask = resize(mask, (mask_h, mask_w))

        outfile = (outfolder / file.relative_to(infolder)).with_suffix(".csv")
        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)

        with open(outfile, "w") as f:
            writer = csv.DictWriter(f, ["id", "parent", "level", "x", "y", "dx", "dy"])
            writer.writeheader()
            for k, (y, x) in enumerate(numpy.argwhere(mask)):
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
