# coding: utf8
"""
A module to extract patches in a slide.

Enable filtering on tissue surface ratio.
Draft for hierarchical patch extraction and representation is proposed.
"""

import os
from .functional_api import patchify_slide
from .functional_api import patchify_folder
from .functional_api import patchify_slide_hierarchically
from .functional_api import patchify_folder_hierarchically
from .filters import standardize_filters


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class UnknownLevelError(Error):
    """
    Raise when no class is found in a datafolder.

    *********************************************
    """

    pass


class Patchifier(object):
    """A class to handle patchification tasks."""

    def __init__(self,
                 outdir,
                 level,
                 psize,
                 interval,
                 offset={"x": 0, "y": 0},
                 filters=[],
                 verbose=2):
        """
        Create the patchifier.

        Arguments:
            - outdir: str, path to an output directory.

        """
        self.outdir = outdir
        self.level = level
        self.psize = psize
        self.interval = interval
        self.offset = offset
        self.filters = filters
        self.verbose = verbose

    def patchify(self, path):
        """
        Patchify a slide or an entire folder.

        Arguments:
            - path: str, absolute path to a slide or a folder of slides.

        """
        if os.path.isdir(path):
            patchify_folder(path, self.outdir, self.level, self.psize, self.interval, offset=self.offset, filters=self.filters, verbose=self.verbose)
        else:
            patchify_slide(path, self.outdir, self.level, self.psize, self.interval, offset=self.offset, filters=self.filters, verbose=self.verbose)

    def add_filter(self, filter_func):
        """
        Add a filter function to this patch extractor.

        Arguments:
            - filter_func: callable, a function that take an image as argument and output a bool.

        """
        self.filters.append(filter_func)


class HierarchicalPatchifier(object):
    """A class to handle patchification tasks."""

    def __init__(self,
                 outdir,
                 top_level,
                 low_level,
                 psize,
                 interval,
                 offset={"x": 0, "y": 0},
                 filters={},
                 silent=[],
                 verbose=2):
        """
        Create the hierarchical patchifier.

        Arguments:
            - outdir: str, path to an output directory.

        """
        self.outdir = outdir
        self.top_level = top_level
        self.low_level = low_level
        self.psize = psize
        self.interval = interval
        self.offset = offset
        self.filters = standardize_filters(filters, top_level, low_level)
        self.verbose = verbose
        self.silent = silent

    def patchify(self, path):
        """
        Patchify hierarchically a slide or an entire folder.

        Arguments:
            - path: str, absolute path to a slide or a folder of slides.

        """
        if os.path.isdir(path):
            patchify_folder_hierarchically(path,
                                           self.outdir,
                                           self.top_level,
                                           self.low_level,
                                           self.psize,
                                           self.interval,
                                           offset=self.offset,
                                           filters=self.filters,
                                           silent=self.silent,
                                           verbose=self.verbose)
        else:
            patchify_slide_hierarchically(path,
                                          self.outdir,
                                          self.top_level,
                                          self.low_level,
                                          self.psize,
                                          self.interval,
                                          offset=self.offset,
                                          filters=self.filters,
                                          silent=self.silent,
                                          verbose=self.verbose)

    def add_filter(self, filter_func, level=None):
        """
        Add a filter function to the hierarchical patch extractor.

        Arguments:
            - filter_func: callable, a function that take an image as argument and output a bool.

        """
        # if level is None, append filter func to all levels
        if level is None:
            for l in self.filters:
                self.filters[l].append(filter_func)
        # if level is int, append filter func to this level
        elif type(level) == int:
            if level not in self.filters:
                raise UnknownLevelError("Level {} is not in {} !!!".format(level, list(self.filters.keys())))
            self.filters[level].append(filter_func)
        elif type(level) == list:
            for l in level:
                self.add_filter(filter_func, l)
        else:
            raise UnknownLevelError("{} is not a valid type of level !!!".format(level))
