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
from ..util.types import (
    Filter,
    FilterList,
    PathLike,
    NDImage,
    NDBoolMask,
    Coord,
    Patch,
    Slide,
)
from .visu import preview_from_queries
from .filters import (
    filter_hasdapi,
    filter_has_significant_dapi,
    filter_has_tissue_he,
    standardize_filters,
)
from .slide_filters import filter_thumbnail, filter_fluo_thumbnail
from .compat import convert_coords
import os
import csv
from skimage.io import imsave
from tqdm import tqdm
from .errors import UnknownFilterError, InvalidArgument
from typing import Optional, Sequence, Tuple, Iterator, Dict
from ..util.convert import (
    get_categorical_segments_from_edges,
    get_categorical_layer_edges,
    layer_segment_to_json_struct,
    gen_categorical_from_floatpreds,
    colorCycle,
)
import json
from fastcore.basics import listify


izi_filters = {
    "has-dapi": filter_hasdapi,
    "has-significant-dapi": filter_has_significant_dapi,
    "has-tissue-he": filter_has_tissue_he,
}

slide_filters = {"full": filter_thumbnail, "fluo": filter_fluo_thumbnail}


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
            if filt not in izi_filters:
                raise UnknownFilterError("{} is not a valid filter !!!".format(filt))
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
            if filt not in slide_filters:
                raise UnknownFilterError("{} is not a valid filter !!!".format(filt))
            filt = slide_filters[filt]
        mask = mask & filt(thumb)
    return mask


def slide_rois(
    slide: Slide,
    level: int,
    psize: Coord,
    interval: Coord = (0, 0),
    ancestors: Optional[Sequence[Patch]] = None,
    offset: Coord = (0, 0),
    filters: Optional[Sequence[Filter]] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
) -> Iterator[Tuple[Patch, NDImage]]:
    """
    Get patches with coordinates.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels, get patches with its coordinates.

    Args:
        slide: the slide to patchify.
        level: pyramid level.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        ancestors: patches that contain upcoming patches.
        offset: (x, y) offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean mask.

    Yields:
        A tuple containing a Patch object and the corresponding image as
        ndarray.

    """
    psize = convert_coords(psize)
    offset = convert_coords(offset)
    ancestors = ifnone(ancestors, [])
    filters = listify(filters)
    slide_filters = listify(slide_filters)
    if len(ancestors) > 0:
        mag = slide.level_downsamples[level]
        shape = Coord(ancestors[0].size_0) / mag
        size_0 = psize * mag
        patches = []
        for ancestor in ancestors:
            # ancestor is a patch
            prefix = ancestor.id
            k = 0
            for patch_coord in regular_grid(shape, interval, psize):
                k += 1
                idx = "{}#{}".format(prefix, k)
                position = patch_coord * mag + ancestor.position
                patches.append(
                    Patch(
                        id=idx,
                        slidename=slide._filename.split("/")[-1],
                        position=position,
                        level=level,
                        size=psize,
                        size_0=size_0,
                        parent=ancestor,
                    )
                )
        for patch in tqdm(patches, ascii=True):
            try:
                image = slide.read_region(patch.position, patch.level, patch.size)
                image = numpy.array(image)[:, :, 0:3]
                if filter_image(image, filters):
                    yield patch, image
            except openslide.lowlevel.OpenSlideError:
                print(
                    "small failure while reading tile x={}, y={} in {}".format(
                        *patch.position, slide._filename
                    )
                )
    else:
        shape = Coord(*slide.level_dimensions[level])
        mag = slide.level_downsamples[level]
        thumb = numpy.array(slide.get_thumbnail((thumb_size, thumb_size)))
        mask = apply_slide_filters(thumb, slide_filters)
        k = 0
        for patch_coord in get_coords_from_mask(mask, shape, interval, psize):
            k += 1
            idx = "#{}".format(k)
            position = patch_coord * mag + offset
            size_0 = psize * mag
            try:
                image = slide.read_region(position, level, psize)
                image = numpy.array(image)[:, :, 0:3]
                if filter_image(image, filters):
                    yield Patch(
                        id=idx,
                        slidename=slide._filename.split("/")[-1],
                        position=position,
                        level=level,
                        size=psize,
                        size_0=size_0,
                    ), image
            except openslide.lowlevel.OpenSlideError:
                print(
                    "small failure while reading tile x={}, y={} in {}".format(
                        *position, slide._filename
                    )
                )


def slide_rois_no_image(
    slide: Slide,
    level: int,
    psize: Coord,
    interval: Coord = (0, 0),
    ancestors: Optional[Sequence[Patch]] = None,
    offset: Coord = (0, 0),
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
) -> Iterator[Patch]:
    """
    Get patches with coordinates.

    Given a slide, a pyramid level, a patchsize in pixels, an interval in pixels
    and an offset in pixels, get patches with its coordinates. Does not export image at
    any point.

    Args:
        slide: the slide to patchify.
        level: pyramid level.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        ancestors: patches that contain upcoming patches.
        offset: (x, y) offset in px on x and y axis for patch start.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean mask.

    Yields:
        A tuple containing a Patch object and the corresponding image as
        ndarray.

    """
    psize = convert_coords(psize)
    offset = convert_coords(offset)
    ancestors = ifnone(ancestors, [])
    slide_filters = listify(slide_filters)
    if len(ancestors) > 0:
        mag = slide.level_downsamples[level]
        shape = Coord(ancestors[0].size_0) / mag
        size_0 = psize * mag
        for ancestor in ancestors:
            # ancestor is a patch
            prefix = ancestor.id
            k = 0
            for patch_coord in regular_grid(shape, interval, psize):
                k += 1
                idx = "{}#{}".format(prefix, k)
                position = patch_coord * mag + ancestor.position
                yield Patch(
                    id=idx,
                    slidename=slide._filename.split("/")[-1],
                    position=position,
                    level=level,
                    size=psize,
                    size_0=size_0,
                    parent=ancestor,
                )
    else:
        shape = Coord(*slide.level_dimensions[level])
        mag = slide.level_downsamples[level]
        thumb = numpy.array(slide.get_thumbnail((thumb_size, thumb_size)))
        mask = apply_slide_filters(thumb, slide_filters)
        k = 0
        for patch_coord in get_coords_from_mask(mask, shape, interval, psize):
            k += 1
            idx = "#{}".format(k)
            position = patch_coord * mag + offset
            size_0 = psize * mag
            yield Patch(
                id=idx,
                slidename=slide._filename.split("/")[-1],
                position=position,
                level=level,
                size=psize,
                size_0=size_0,
            )


def patchify_slide(
    slidefile: PathLike,
    outdir: PathLike,
    level: int,
    psize: Coord,
    interval: Coord = (0, 0),
    offset: Coord = (0, 0),
    filters: Optional[Sequence[Filter]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
    silence: int = 0,
    backend: str = "openslide",
):
    """
    Save patches of a given wsi.

    Args:
        slidefile: abs path to slide file.
        outdir: abs path to an output folder.
        level: pyramid level.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        offset: (x, y) offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean
            mask.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of
            processes, thumbnail export.
        silence: 0 => write images, 1 => use images, but do not write, 2 => not even
            using images for filtering.
        backend: whether to use openslide or cucim as backend.

    """
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
    slide = Slide(slidefile, backend=backend)
    plist = []
    # level directory
    outleveldir = os.path.join(slide_folder_output, "level_{}".format(level))
    if os.path.isdir(outleveldir):
        safe_rmtree(outleveldir, ignore_errors=True, erase_tree=erase_tree)
    os.makedirs(outleveldir, exist_ok=True)
    ########################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if silence == 0:
            # Write images
            for patch, img in slide_rois(
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
                    outleveldir, "{}_{}_{}.png".format(*patch.position, patch.level)
                )
                imsave(outfile, img)
                plist.append(patch)
        elif silence == 1:
            # Do not write patches (but use images to filter)
            for patch, img in slide_rois(
                slide,
                level,
                psize,
                interval,
                offset=offset,
                filters=filters,
                thumb_size=thumb_size,
                slide_filters=slide_filters,
            ):
                plist.append(patch)
        elif silence == 2:
            # Do not write images and do not even use them to filter patches
            for patch in slide_rois_no_image(
                slide,
                level,
                psize,
                interval,
                offset=offset,
                thumb_size=thumb_size,
                slide_filters=slide_filters,
            ):
                plist.append(patch)
        else:
            raise InvalidArgument(
                "Invalid 'silence' parameter: '{}',"
                " should be one of [0, 1, 2]".format(silence)
            )
    if verbose > 1:
        print("end of patchification.")
        print("starting metadata csv export...")
    csv_columns = Patch.get_fields()
    csv_path = os.path.join(slide_folder_output, "patches.csv")
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        writer.writerows(map(Patch.to_csv_row, plist))
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
    psize: Coord,
    interval: Coord = (0, 0),
    offset: Coord = (0, 0),
    filters: Optional[FilterList] = None,
    silent: Optional[Sequence[int]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
    backend: str = "openslide",
):
    """
    Save patches of a given wsi in a hierarchical way.

    Args:
        slidefile: abs path to a slide file.
        outdir: abs path to an output folder.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        offset: (x, y) offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        silent: pyramid level not to output.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        thumb_size: size of thumbnail's longest side. Always preserves aspect ratio.
        slide_filters: list of filters to apply to thumbnail. Should output boolean
            mask.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.
        backend: whether to use openslide or cucim as backend.

    """
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

    csv_columns = Patch.get_fields()
    csv_path = os.path.join(slide_folder_output, "patches.csv")

    slide = Slide(slidefile, backend=backend)

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
                for patch, img in slide_rois(
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
                            "{}_{}_{}.png".format(*patch.position, patch.level),
                        )
                        imsave(outfile, img)
                    current_plist.append(patch)
            plist = [p for p in current_plist]
            if verbose > 1:
                print("end of patchification.")
                print("starting metadata csv export...")
            writer.writerows(map(Patch.to_csv_row, plist))
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
    psize: Coord,
    interval: Coord = (0, 0),
    offset: Coord = (0, 0),
    filters: Optional[Sequence[Filter]] = None,
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = False,
    folders: Optional[Sequence[str]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
    silence: int = 0,
    backend: str = "openslide",
):
    """
    Save patches of all wsi inside a folder.

    Args:
        infolder: abs path to a folder of slides.
        outfolder: abs path to an output folder.
        level: pyramid level.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        offset: (x, y) offset in px on x and y axis for patch start.
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
        silence: 0 => write images, 1 => use images, but do not write, 2 => not even
            using images for filtering.
        backend: whether to use openslide or cucim as backend.

    """
    if os.path.isdir(outfolder):
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
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
                silence=silence,
                backend=backend,
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
    psize: Coord,
    interval: Coord = (0, 0),
    offset: Coord = (0, 0),
    filters: Optional[FilterList] = None,
    silent: Optional[Sequence[int]] = None,
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = False,
    folders: Optional[Sequence[str]] = None,
    erase_tree: Optional[bool] = None,
    thumb_size: int = 512,
    slide_filters: Optional[Sequence[Filter]] = None,
    verbose: int = 2,
    backend: str = "openslide",
):
    """
    Save hierarchical patches of all wsi inside a folder.

    Args:
        infolder: abs path to a folder of slides.
        outfolder: abs path to an output folder.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.
        psize: (w, h) size of the patches (in pixels).
        interval: (x, y) interval between 2 neighboring patches.
        offset: (x, y) offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        silent: pyramid level not to output.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        recurse: whether to look for files recursively.
        folders: list of subfolders to explore when recurse is True. Defaults to all.
        erase_tree: whether to erase outfolder if it exists. If None, user will be
            prompted for a choice.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.
        backend: whether to use openslide or cucim as backend.

    """
    if os.path.isdir(outfolder):
        erase_tree = safe_rmtree(outfolder, ignore_errors=True, erase_tree=erase_tree)
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
                backend=backend,
            )
        except (
            openslide.OpenSlideUnsupportedFormatError,
            openslide.lowlevel.OpenSlideError,
        ) as e:
            warnings.warn(str(e))


def export_floatpred_to_categorical_micromap_json(
    pathaiafolder: PathLike,
    slidefolder: PathLike,
    jsonfolder: PathLike,
    level: int,
    task: str,
    thresholds: Dict[int, Tuple[float, float]],
    classnames: Dict[int, str],
    extensions: Sequence[str] = (".mrxs",),
    recurse: bool = True,
):
    """
    Export pathaia csv to a json annotation file compatible with MicroMap.

    For each predicted category, a layer is created and connected components
    are computed.

    Args:
        infolder: abs path to a folder of pathaia csv.
        slidefolder: abs path to a folder of slides.
        jsonfolder: abs path to a folder of json annotations.
        level: level of patches to export.
        task: prediction task to export (csv column).
        thresholds: histogram bins for each category.
        classnames: class names for each category.
        extensions: slide extension.

    """
    color_dict = {k: colorCycle[k] for k in classnames}
    slidefiles = get_files(slidefolder, extensions=extensions, recurse=recurse).map(str)
    for slidefile in tqdm(slidefiles):
        slidename, _ = os.path.splitext(os.path.basename(slidefile))
        slidedir = os.path.dirname(slidefile)
        pathaia_slidedir = slidedir.replace(slidefolder, pathaiafolder)
        pathaiacsv = os.path.join(pathaia_slidedir, slidename, "patches.csv")
        json_slidedir = slidedir.replace(slidefolder, jsonfolder)
        pathaiajson = os.path.join(json_slidedir, "{}.json".format(slidename))
        try:
            slide = openslide.OpenSlide(slidefile)
            # print(f'Saving to json {pathaiajson} from {pathaiacsv}')
            gen = gen_categorical_from_floatpreds(pathaiacsv, level, task, thresholds)
            layer_edges, layer_meta, interval = get_categorical_layer_edges(
                gen, color_dict, classnames
            )
            layer_segments = get_categorical_segments_from_edges(layer_edges)
            annotations = layer_segment_to_json_struct(
                interval, layer_segments, layer_meta, slide
            )
            with open(pathaiajson, "w") as f:
                json.dump(annotations, f)
        except Exception as e:
            warnings.warn(
                "slide '{}'"
                " with patch file '{}'"
                " failed with error: '{}'".format(slidefile, pathaiacsv, str(e))
            )
