# coding: utf8
"""Useful functions for handling patches in WSIs."""

import os


def slides_in_folder(folder, extensions=[".mrxs"]):
    """
    Return slide files inside a folder for a given extension.

    Arguments:
        - folder: absolute path to a directory containing slides.
        - extension: list of str, file extensions of the slides.
    Returns:
        - abspathlist: list of absolute path of slide files.

    """
    abspathlist = []
    for name in os.listdir(folder):

        if name[0] != '.':
            for extension in extensions:
                if name.endswith(extension):
                    abspathlist.append(os.path.join(folder, name))
    return abspathlist


def slide_basename(slidepath):
    """
    Give the basename of a slide from its absolutepath.

    Arguments:
        - slidepath: str, absolute path to a slide.

    Returns:
        - basename: str, basename of the slide.

    """
    base = os.path.basename(slidepath)
    basename, ext = os.path.splitext(base)
    return basename
