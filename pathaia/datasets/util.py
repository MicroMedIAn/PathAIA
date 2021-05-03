# coding: utf8
"""
A module to implement useful functions to apply to dataset.

I still don't knwo exactly what we are putting into this module.
"""
from typing import Sequence, Dict, Any, Optional, Callable, Generator
from ..util.types import RefDataSet
import numpy as np
from .errors import UnknownSplitModeError


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


def clip_dataset(dataset: RefDataSet, max_spl: int) -> RefDataSet:
    """
    Clip a dataset (to a max number of samples).

    Args:
        dataset: samples of a dataset.
        max_spl: max number of samples.

    Returns:
        Clipped dataset.

    """
    x, y = dataset
    mx = min(max_spl, len(dataset[0]))
    return x[0:mx], y[0:mx]


def split_dataset(
    dataset: RefDataSet,
    val_ratio: float,
    test_ratio: Optional[float],
    mode: str = "tv"
) -> Dict[str, RefDataSet]:
    """
    Compute split of the dataset from validation ratio and split mode.

    Args:
        dataset: samples of a dataset.
        val_ratio: ratio of validation data.
        mode: split mode, one of 'tv' (training-validation) or 'tvt' (training-validation-test)

    Returns:
        train, validation (test optional) datasets.

    """
    x, y = dataset
    size = len(x)
    val_size = int(val_ratio * size)
    if mode == "tv":
        x_val = x[0:val_size]
        y_val = y[0:val_size]
        x_train = x[val_size::]
        y_train = y[val_size::]
        return {
            "training": (x_train, y_train),
            "validation": (x_val, y_val)
        }
    if mode == "tvt":
        test_size = int(test_ratio * size)
        x_val = x[0:val_size]
        y_val = y[0:val_size]

        start_test = val_size
        end_test = val_size + test_size
        x_test = x[start_test:end_test]
        y_test = y[start_test:end_test]

        start_train = end_test
        x_train = x[start_train::]
        y_train = y[start_train::]

        return {
            "training": (x_train, y_train),
            "validation": (x_val, y_val),
            "test": (x_test, y_test)
        }
    raise UnknownSplitModeError(
        "{} is not a valid split mode! It should be either 'tv' or 'tvt'!".format(mode)
    )


# Decorators

# Careful here, since above functions are used as pre-processing steps,
# (called before the wrapped function)
# the calling order of the decorators is reversed:
# ---------
# @shuffle
# @clip
# def my_generator(dataset)
# -------------------------
# will first shuffle, then clip the dataset...

# Yet, some decorator apply directly to the generator,
# like 'batch' for example:
# ---------
# @batch(2)
# @shuffle
# @clip
# def my_generator(dataset)
# -------------------------
# will first shuffle, then clip the dataset, then put samples in batches of 2

def shuffle(data_generator: Callable) -> Callable:
    """
    Decorate a data generator function with the shuffle function.

    Args:
        data_generator: a function that takes a dataset and yield samples.

    Returns:
        shuffle the dataset before the data_generator is applied.

    """
    def shuffled_version(dataset: RefDataSet) -> Generator:
        """
        Wrap the data_generator in this function.

        Args:
            dataset: just a dataset.

        Returns:
            shuffled version of the data generator.

        """
        return data_generator(shuffle_dataset(dataset))
    return shuffled_version


def balance(data_generator: Callable) -> Callable:
    """
    Decorate a data generator function with the balance function.

    Args:
        data_generator: a function that takes a dataset and yield samples.

    Returns:
        balance the dataset before the data_generator is applied.

    """
    def balanced_version(dataset: RefDataSet) -> Generator:
        """
        Wrap the data_generator in this function.

        Args:
            dataset: just a dataset.

        Returns:
            balanced version of the data generator.

        """
        return data_generator(balance_dataset(dataset))
    return balanced_version


def clip(max_spl: int) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clip function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clip the dataset before the data_generator is applied.

        """
        def clipped_version(dataset: RefDataSet) -> Generator:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                clipped version of the data generator.

            """
            return data_generator(clip_dataset(dataset))
        return clipped_version
    return decorator


def batch(batch_size: int, keep_last: bool = False) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the batch function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            batch the dataset before the data_generator is applied.

        """
        def batched_version(dataset: RefDataSet) -> Generator:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                batch version of the data generator.

            """
            xb = []
            yb = []
            gen = data_generator(dataset)
            for x, y in gen:
                if len(xb) == batch_size:
                    xb = []
                    yb = []
                xb.append(x)
                yb.append(y)
                if len(xb) == batch_size:
                    yield xb, yb
            if len(xb) > 0 and len(xb) < batch_size and keep_last:
                yield xb, yb
        return batched_version
    return decorator


def clean(dtype: type, rm: Sequence[Any]) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clean function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def cleaned_version(dataset: RefDataSet) -> Generator:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            return data_generator(clean_dataset(dataset, dtype, rm))
        return cleaned_version
    return decorator
