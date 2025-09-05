# coding: utf8
"""Useful functions for handling patches in WSIs."""

import os
import numpy
from pathlib import Path
from fastcore.foundation import L, setify
import shutil
from typing import Sequence, List, Optional, Dict
from .types import PathLike


def slides_in_folder(folder: str, extensions: Sequence[str] = (".mrxs",)) -> List[str]:
    """
    Return slide files inside a folder for a given extension.

    Args:
        folder: absolute path to a directory containing slides.
        extension: file extensions of the slides.
    Returns:
        List of absolute paths of slide files.

    """
    abspathlist = []
    for name in os.listdir(folder):

        if not name.startswith("."):
            for extension in extensions:
                if name.endswith(extension):
                    abspathlist.append(os.path.join(folder, name))
    return abspathlist


def slide_basename(slidepath: str) -> str:
    """
    Give the basename of a slide from its absolutepath.

    Args:
        slidepath: absolute path to a slide.

    Returns:
        basename: basename of the slide.

    """
    base = os.path.basename(slidepath)
    basename, ext = os.path.splitext(base)
    return basename


def imfiles_in_folder(
    folder: str,
    authorized: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    forbiden: Sequence[str] = ("thumbnail",),
    randomize: bool = False,
    datalim: Optional[int] = None,
) -> List[str]:
    """
    Get image files in a given folder.

    Get all image files (selected by file extension). You can remove terms
    from the research.

    Args:
        folder: absolute path to an image directory.
        authorized: authorized image file extensions.
        forbiden: non-authorized words in file names.
        randomize: whether to randomize output list of files.
        datalim: maximum number of file to extract in folder.

    Returns:
        Absolute paths of image files in folder.

    """
    imfiles = []
    for name in os.listdir(folder):
        _, ext = os.path.splitext(name)
        if ext in authorized:
            auth = True
            for forb in forbiden:
                if forb in name:
                    auth = False
            if auth:
                imfiles.append(os.path.join(folder, name))

    if randomize:
        numpy.random.shuffle(imfiles)
    if datalim is not None:
        imfiles = imfiles[0:datalim]

    return imfiles


def dataset2folders(
    projfolder: PathLike,
    level: int,
    randomize: bool = False,
    slide_data_lim: Optional[int] = None,
) -> Dict[str, str]:
    """
    Link slidenames to their pathaia patch folder.

    A pathaia patch folder is named with an int corresponding
    to the level of patch extraction.

    Args:
        projfolder: absolute path to a pathaia project folder.
        level: pyramid level of patch extraction to consider.
        randomize: whether to randomize output list of slides.
        slide_data_lim: number of slides to consider in project.

    Returns:
        Dictionary mapping slidenames and absolute paths to patch dirs.

    """
    slide2folder = dict()
    for slidename in os.listdir(projfolder):
        slide_folder = os.path.join(projfolder, slidename)
        if os.path.isdir(slide_folder):
            level_folder = os.path.join(slide_folder, "level_{}".format(level))
            if os.path.isdir(level_folder):
                slide2folder[slidename] = level_folder

    keep = list(slide2folder.keys())
    if randomize:
        numpy.random.shuffle(keep)
    if slide_data_lim is not None:
        keep = keep[0:slide_data_lim]
    return {k: slide2folder[k] for k in keep}


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path: PathLike,
    extensions: Optional[Sequence[str]] = None,
    recurse: bool = True,
    folders: Optional[Sequence[str]] = None,
    followlinks: bool = True,
) -> List[Path]:
    """
    Find all files in a folder recursively.

    Arguments:
        path: Path to input folder.
        extensions: list of acceptable file extensions.
        recurse: whether to perform a recursive search or not.
        folders: direct subfolders to explore (if None explore all).
        followlinks: whether to follow symlinks or not.

    Returns:
        List of all absolute paths to found files.
    """
    path = Path(path)
    folders = L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)


def safe_rmtree(
    path: PathLike, ignore_errors: bool = True, erase_tree: Optional[bool] = None
) -> bool:
    """
    Safe version of rmtree that asks for permission before deleting.

    Arguments:
        path: path to folder to be deleted.
        ignore_error: whether to ignore errors or not.
        erase_tree: whether to remove tree or not. If None asks for permission.

    Returns:
        True if erase_tree==True or if user gave permission.
    """
    response = ""
    if erase_tree is None:
        while response not in ["y", "n"]:
            response = input(
                "Are you sure you want to delete " f"{path} and all subfolders ? y/n"
            )
            response = response.lower()

    if response == "y" or erase_tree:
        shutil.rmtree(path, ignore_errors=ignore_errors)
        return True
    else:
        return False
