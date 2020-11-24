# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""

import os
from ..util.basic import ifnone
from .functional_api import patchify_slide
from .functional_api import patchify_folder
from .functional_api import patchify_slide_hierarchically
from .functional_api import patchify_folder_hierarchically
from .filters import standardize_filters
from .errors import UnknownLevelError


class Patchifier(object):
    """A class to handle patchification tasks."""

    def __init__(
        self, outdir, level, psize, interval, offset=None, filters=None, verbose=2
    ):
        """
        Create the patchifier.

        Args:
            outdir (str): path to an output directory.
            level (int): pyramid level to extract.
            psize (int): size of the side of the patches (in pixels).
            interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
            offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
            filters (list of func): filters to accept patches.
            verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

        """
        self.outdir = outdir
        self.level = level
        self.psize = psize
        self.interval = interval
        self.offset = ifnone(offset, {"x": 0, "y": 0})
        self.filters = ifnone(filters, [])
        self.verbose = verbose

    def patchify(self, path):
        """
        Patchify a slide or an entire folder of slides.

        Args:
            path (str): absolute path to a slide or a folder of slides.

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

    def add_filter(self, filter_func):
        """
        Add a filter function to this patch extractor.

        Args:
            filter_func (callable): a function that take an image as argument and output a bool.

        """
        self.filters.append(filter_func)


class HierarchicalPatchifier(object):
    """A class to handle patchification tasks."""

    def __init__(
        self,
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
        Create the hierarchical patchifier.

        Args:
            outdir (str): path to an output directory.
            top_level (int): top pyramid level to consider.
            low_level (int): lowest pyramid level to consider.
            psize (int): size of the side of the patches (in pixels).
            interval (dictionary): {"x", "y"} interval between 2 neighboring patches.
            offset (dictionary): {"x", "y"} offset in px on x and y axis for patch start.
            filters (dict of list of func): filters to accept patches.
            silent (list of int): pyramid level not to output.
            verbose (int): 0 => nada, 1 => patchifying parameters, 2 => start-end of processes, thumbnail export.

        """
        self.outdir = outdir
        self.top_level = top_level
        self.low_level = low_level
        self.psize = psize
        self.interval = interval
        self.offset = ifnone(offset, {"x": 0, "y": 0})
        self.filters = standardize_filters(ifnone(filters, {}), top_level, low_level)
        self.verbose = verbose
        self.silent = ifnone(silent, [])

    def patchify(self, path):
        """
        Patchify hierarchically a slide or an entire folder of slides.

        Args:
            path (str): absolute path to a slide or a folder of slides.

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

    def add_filter(self, filter_func, level=None):
        """
        Add a filter function to the hierarchical patch extractor.

        Args:
            filter_func (callable): a function that take an image as argument and output a bool.

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
