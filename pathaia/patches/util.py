# coding: utf8
"""Useful function for patch extraction."""
import numpy
import os
import itertools


def log_magnification(slide, level):
    return int(numpy.log2(slide.level_dimensions[0][0] / slide.level_dimensions[level][0]))


def magnification(slide, level):
    return int(slide.level_dimensions[0][0] / slide.level_dimensions[level][0])



def iterpatches(slide, rois):

    for patch in rois:
        istart = patch[0]
        jstart = patch[2]
        di = patch[1] - patch[0]
        dj = patch[3] - patch[2]

        image = slide.read_region((jstart, istart), 0, (dj, di))
        image = numpy.array(image)[:, :, 0:3]

        yield patch, image


def iterpatches_at_level(slide, level, rois):

    for patch in rois:
        istart = patch[0]
        jstart = patch[2]
        di = patch[1] - patch[0]
        dj = patch[3] - patch[2]

        image = slide.read_region((jstart, istart), level, (dj, di))
        image = numpy.array(image)[:, :, 0:3]

        yield patch, image


def listpatches(slide, rois):
    lp = []
    li = []

    for patch in rois:
        istart = patch[0]
        jstart = patch[2]
        di = patch[1] - patch[0]
        dj = patch[3] - patch[2]

        image = slide.read_region((jstart, istart), 0, (dj, di))
        image = numpy.array(image)[:, :, 0:3]
        li.append(image)
        lp.append(patch)

    return lp, li


def slides_in_folder(folder):

    abspathlist = []

    for name in os.listdir(folder):

        if name[0] != '.' and '.mrxs' in name:

            abspathlist.append(os.path.join(folder, name))

    return abspathlist


def slide_basename(slidepath):

    base = os.path.basename(slidepath)
    slidebasename = base[0:-len('.mrxs')]
    return slidebasename


def regular_grid(shape, width):
    maxi = width * int(shape[0] / width)
    maxj = width * int(shape[1] / width)
    col = numpy.arange(start=0, stop=maxj, step=width, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=width, dtype=int)
    for p in itertools.product(line, col):
        yield p
