# coding: utf8
"""
A module to implement useful functions to apply to dataset.

I still don't knwo exactly what we are putting into this module.
"""
from typing import Sequence, Dict, Any, Callable, Generator
from ..util.types import RefDataSet
import numpy as np


def info(dataset: RefDataSet) -> Dict:
    """
    Check multiple filters on an image.

    Args:
        dataset: samples of a dataset.

    Returns:
        Unique labels in the dataset with associated population.

    """
    x, y = dataset
    info = dict()
    for spl in y:
        if spl not in info:
            info[spl] = 1
        else:
            info[spl] += 1
    return info


def shuffle_dataset(dataset: RefDataSet) -> RefDataSet:
    """
    Shuffle samples in a dataset.

    Args:
        dataset: samples of a dataset.

    Returns:
        Shuffled dataset.

    """
    x, y = dataset
    ridx = np.arange(len(x))
    np.random.shuffle(ridx)
    rx = [x[i] for i in ridx]
    ry = [y[i] for i in ridx]
    return rx, ry


def clean_dataset(dataset: RefDataSet, dtype: type, rm: Sequence[Any]) -> RefDataSet:
    """
    Remove bad data from a reference dataset.

    Args:
        dataset: samples of a dataset.
        dtype: type of data to keep.
        rm: sequence of labels to remove from the dataset.

    Returns:
        Purified dataset.

    """
    x, y = dataset
    pure_x = []
    pure_y = []
    for spl_x, spl_y in zip(x, y):
        if isinstance(spl_y, dtype) and spl_y not in rm:
            pure_x.append(spl_x)
            pure_y.append(spl_y)
    return pure_x, pure_y


def balance_cat(dataset: RefDataSet, cat: Any, lack: int) -> RefDataSet:
    """
    Compensate lack of a category in a dataset by random sample duplication.

    Args:
        dataset: samples of a dataset.
        cat: label in the dataset to enrich.
        missing: missing samples in the dataset to reach expected population.

    Returns:
        Balanced category.

    """
    x, y = dataset
    cat_x = [spl for spl, lab in zip(x, y) if lab == cat]
    ridx = np.arange(len(cat_x))
    np.random.shuffle(ridx)
    ridx = ridx[0:lack]
    x_padding = [cat_x[rd] for rd in ridx]
    y_padding = [cat for k in range(lack)]
    new_x = [spl for spl in x] + x_padding
    new_y = [spl for spl in y] + y_padding
    return new_x, new_y


def balance_dataset(dataset: RefDataSet) -> RefDataSet:
    """
    Balance the dataset using the balance_cat function on each cat.

    Args:
        dataset: samples of a dataset.

    Returns:
        The balanced dataset.

    """
    x = []
    y = []
    cat_count = info(dataset)
    maj_count = max(cat_count.values())
    for cat, count in cat_count.items():
        lack = maj_count - count
        if lack > 0:
            x, y = balance_cat(dataset, cat, lack)
    return x, y


def fair_dataset(dataset: RefDataSet, dtype: type, rm: Sequence[Any]) -> RefDataSet:
    """
    Make a dataset fair.

    Purify, balance and shuffle a dataset.

    Args:
        dataset: samples of a dataset.
        dtype: type of data to keep.
        rm: sequence of labels to remove from the dataset.

    Returns:
        Fair dataset.

    """
    return shuffle_dataset(balance_dataset(clean_dataset(dataset, dtype, rm)))
