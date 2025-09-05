# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""

import os
from typing import Optional, Sequence
from ..util.basic import ifnone
from ..util.types import PathLike, Filter, FilterList, Coord
from .functional_api import patchify_slide
from .functional_api import patchify_folder
from .functional_api import patchify_slide_hierarchically
from .functional_api import patchify_folder_hierarchically
from .filters import standardize_filters
from .errors import UnknownLevelError
from .compat import convert_coords


class Patchifier(object):
    """
    A class to handle patchification tasks.

    Args:
        outdir: path to an output directory.
        level: pyramid level to extract.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes,
            thumbnail export.
    """

    def __init__(
        self,
        outdir: PathLike,
        level: int,
        psize: int,
        interval: Coord,
        offset: Coord = (0, 0),
        filters: Optional[Sequence[Filter]] = None,
        extensions: Optional[Sequence[str]] = None,
        verbose: int = 2,
    ):
        self.outdir = outdir
        self.level = level
        self.psize = psize
        self.interval = convert_coords(interval)
        self.offset = convert_coords(offset)
        self.filters = ifnone(filters, [])
        self.verbose = verbose
        self.extensions = ifnone(extensions, (".mrxs",))

    def patchify(self, path: PathLike):
        """
        Patchify a slide or an entire folder of slides.

        Args:
            path: absolute path to a slide or a folder of slides.

        """
        if os.path.isdir(path):
            patchify_folder(
                path,
                self.outdir,
                self.level,
                self.psize,
                self.interval,
                offset=self.offset,
                filters=self.filters,
                verbose=self.verbose,
                extensions=self.extensions,
            )
        else:
            patchify_slide(
                path,
                self.outdir,
                self.level,
                self.psize,
                self.interval,
                offset=self.offset,
                filters=self.filters,
                verbose=self.verbose,
            )

    def add_filter(self, filter_func: Filter):
        """
        Add a filter function to this patch extractor.

        Args:
            filter_func: a function that take an image as argument and output a bool.

        """
        self.filters.append(filter_func)


class HierarchicalPatchifier(object):
    """
    A class to handle hierachical patchification tasks.

    Args:
        outdir: path to an output directory.
        top_level: top pyramid level to consider.
        low_level: lowest pyramid level to consider.
        psize: size of the side of the patches (in pixels).
        interval: {"x", "y"} interval between 2 neighboring patches.
        offset: {"x", "y"} offset in px on x and y axis for patch start.
        filters: filters to accept patches.
        silent: pyramid level not to output.
        extensions: list of file extensions to consider. Defaults to '.mrxs'.
        verbose: 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

    """

    def __init__(
        self,
        outdir: PathLike,
        top_level: int,
        low_level: int,
        psize: int,
        interval: Coord,
        offset: Coord = (0, 0),
        filters: Optional[FilterList] = None,
        silent: Optional[Sequence[int]] = None,
        extensions: Optional[Sequence[str]] = None,
        verbose: int = 2,
    ):
        self.outdir = outdir
        self.top_level = top_level
        self.low_level = low_level
        self.psize = psize
        self.interval = convert_coords(interval)
        self.offset = convert_coords(offset)
        self.filters = standardize_filters(ifnone(filters, {}), top_level, low_level)
        self.verbose = verbose
        self.silent = ifnone(silent, [])
        self.extensions = ifnone(extensions, (".mrxs",))

    def patchify(self, path: PathLike):
        """
        Patchify hierarchically a slide or an entire folder of slides.

        Args:
            path: absolute path to a slide or a folder of slides.

        """
        if os.path.isdir(path):
            patchify_folder_hierarchically(
                path,
                self.outdir,
                self.top_level,
                self.low_level,
                self.psize,
                self.interval,
                offset=self.offset,
                filters=self.filters,
                silent=self.silent,
                extensions=self.extensions,
                verbose=self.verbose,
            )
        else:
            patchify_slide_hierarchically(
                path,
                self.outdir,
                self.top_level,
                self.low_level,
                self.psize,
                self.interval,
                offset=self.offset,
                filters=self.filters,
                silent=self.silent,
                verbose=self.verbose,
            )

    def add_filter(self, filter_func: Filter, level: Optional[int] = None):
        """
        Add a filter function to the hierarchical patch extractor.

        Args:
            filter_func: a function that take an image as argument and output a bool.

        """
        # if level is None, append filter func to all levels
        if level is None:
            for lev in self.filters:
                self.filters[lev].append(filter_func)
        # if level is int, append filter func to this level
        elif type(level) == int:
            if level not in self.filters:
                raise UnknownLevelError(
                    "Level {} is not in {} !!!".format(level, list(self.filters.keys()))
                )
            self.filters[level].append(filter_func)
        elif type(level) == list:
            for lev in level:
                self.add_filter(filter_func, lev)
        else:
            raise UnknownLevelError("{} is not a valid type of level !!!".format(level))
