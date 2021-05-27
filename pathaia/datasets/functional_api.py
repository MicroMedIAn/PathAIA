# coding: utf8
"""
A module to implement useful functions to apply to dataset.

I still don't knwo exactly what we are putting into this module.
"""
from typing import (
    Sequence, Dict, Any, Callable, Generator, Union, Iterable
)
from ..util.types import RefDataSet, SplitDataSet, DataSet
import numpy as np
from .errors import (
    InvalidDatasetError,
    InvalidSplitError,
    TagNotFoundError
)
import openslide
from .data import fast_slide_query


def extend_to_split_datasets(processing: Callable) -> Callable:
    """
    Decorate a dataset processing to extend usage to split datasets.

    Args:
        processing: a function that takes a RefDataSet and return a RefDataSet.

    Returns:
        Function adapted to Dataset inputs.

    """
    def extended_version(
        dataset: DataSet, *args, **kwargs
    ) -> Union[Dict, DataSet]:
        """
        Wrap the processing in this function.

        Args:
            dataset: just a dataset.

        Returns:
            shuffled version of the data generator.

        """
        if isinstance(dataset, tuple):
            return processing(dataset, *args, **kwargs)
        if isinstance(dataset, dict):
            result = dict()
            for set_name, set_data in dataset.items():
                result[set_name] = processing(set_data, *args, **kwargs)
            return result
        raise InvalidDatasetError(
            "{} is not a valid type for datasets!"
            " It should be a {}...".format(type(dataset), DataSet)
        )
    return extended_version


@extend_to_split_datasets
def info(dataset: RefDataSet) -> Dict:
    """
    Produce info on an unsplitted dataset.

    Args:
        dataset: samples of a dataset.

    Returns:
        Unique labels in the dataset with associated population.

    """
    x, y = dataset
    info = dict()
    for tag in y:
        if tag not in info:
            info[tag] = 1
        else:
            info[tag] += 1
    return info


@extend_to_split_datasets
def ratio_info(dataset: RefDataSet) -> Dict:
    """
    Produce ratios info on an unsplitted dataset.

    Args:
        dataset: samples of a dataset.

    Returns:
        Unique labels in the dataset with associated population.

    """
    x, y = dataset
    populations = dict()
    result = dict()
    for tag in y:
        if tag not in populations:
            populations[tag] = 1
        else:
            populations[tag] += 1
    for tag, population in populations.items():
        result[tag] = float(population) / len(y)
    return result


@extend_to_split_datasets
def class_data(dataset: RefDataSet, class_name: Union[str, int]) -> Dict:
    """
    Produce info on an unsplitted dataset.

    Args:
        dataset: samples of a dataset.

    Returns:
        Unique labels in the dataset with associated population.

    """
    x, y = dataset
    res_x = []
    res_y = []
    if class_name in y:
        for spl, tag in zip(x, y):
            if tag == class_name:
                res_x.append(spl)
                res_y.append(tag)
        return res_x, res_y
    raise TagNotFoundError(
        "Tag '{}' is not in dataset {}!".format(
            class_name, info(dataset)
        )
    )


@extend_to_split_datasets
def shuffle_dataset(dataset: RefDataSet) -> RefDataSet:
    """
    Shuffle samples in a dataset.

    Args:
        dataset: samples of a dataset.

    Returns:
        Shuffled dataset.

    """
    x, y = dataset
    ridx = np.arange(len(y))
    np.random.shuffle(ridx)
    rx = [x[i] for i in ridx]
    ry = [y[i] for i in ridx]
    return rx, ry


@extend_to_split_datasets
def clean_dataset(
    dataset: RefDataSet, dtype: type, rm: Sequence[Any]
) -> RefDataSet:
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
    x_padding = [cat_x[ridx[k % len(ridx)]] for k in range(lack)]
    y_padding = [cat for k in range(lack)]
    return x_padding, y_padding


@extend_to_split_datasets
def balance_dataset(dataset: RefDataSet) -> RefDataSet:
    """
    Balance the dataset using the balance_cat function on each cat.

    Args:
        dataset: samples of a dataset.

    Returns:
        The balanced dataset.

    """
    x = [xd for xd in dataset[0]]
    y = [yd for yd in dataset[1]]
    cat_count = info(dataset)
    try:
        maj_count = max(cat_count.values())
        for cat, count in cat_count.items():
            lack = maj_count - count
            if lack > 0:
                x_pad, y_pad = balance_cat(dataset, cat, lack)
                x += x_pad
                y += y_pad
        return x, y
    except ValueError as e:
        raise InvalidDatasetError(
            "{} check your dataset: {}".format(e, cat_count)
        )


@extend_to_split_datasets
def fair_dataset(
    dataset: RefDataSet, dtype: type, rm: Sequence[Any]
) -> RefDataSet:
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


@extend_to_split_datasets
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
    sections: Sequence,
    preserve_ratio: bool = True
) -> SplitDataSet:
    """
    Compute split of the dataset from ratios.

    Args:
        dataset: samples of a dataset.
        sections: ratios of different splits, should sum to 1.

    Returns:
        splits of the dataset.

    """
    x, y = dataset
    ratios = ratio_info(dataset)
    population = info(dataset)
    result = dict()

    if isinstance(sections, dict):
        if sum(sections.values()) == 1:
            offsets = {k: 0 for k in ratios.keys()}
            for set_name, set_ratio in sections.items():
                x_set = []
                y_set = []
                for class_name in offsets.keys():
                    offset = offsets[class_name]
                    class_size = population[class_name]
                    class_set_size = int(set_ratio * class_size)
                    cx, cy = class_data(dataset, class_name)
                    cx_set = cx[offset:offset + class_set_size]
                    cy_set = cy[offset:offset + class_set_size]
                    x_set += cx_set
                    y_set += cy_set
                    offsets[class_name] += class_set_size
                result[set_name] = (x_set, y_set)
            return result

        raise InvalidSplitError(
            "Split values provided do not sum to 1: {}".format(sections)
        )

    if isinstance(sections, list) or isinstance(sections, tuple):
        if sum(sections) == 1:
            offsets = {k: 0 for k in ratios.keys()}
            for set_name, set_ratio in enumerate(sections):
                x_set = []
                y_set = []
                for class_name in offsets.keys():
                    offset = offsets[class_name]
                    class_size = population[class_name]
                    class_set_size = int(set_ratio * class_size)
                    cx, cy = class_data(dataset, class_name)
                    cx_set = cx[offset:offset + class_set_size]
                    cy_set = cy[offset:offset + class_set_size]
                    x_set += cx_set
                    y_set += cy_set
                    offsets[class_name] += class_set_size
                result[set_name] = (x_set, y_set)
            return result

        raise InvalidSplitError(
            "Split values provided do not sum to 1: {}".format(sections)
        )

    raise InvalidSplitError(
        "Invalid arguments provided to the split method: \n{}\n{}".format(
            sections, info(dataset)
        )
    )


# Decorators on dataset generators

# Careful here, since above functions are used as pre-processing steps,
# (called before the wrapped function)
# the calling order of the decorators is reversed:
# ---------
# @clean   -|
# @balance -|-----> @be_fair
# @shuffle -|
# @clip
# @split
# @batch
# def my_generator(dataset):
#   x, y = dataset
#   for sx, sy in zip(x, y):
#       yield sx, sy
# -------------------------
# will first shuffle, then clip the dataset...

def pre_shuffle(data_generator: Callable) -> Callable:
    """
    Decorate a data generator function with the shuffle function.

    Args:
        data_generator: a function that takes a dataset and yield samples.

    Returns:
        shuffle the dataset before the data_generator is applied.

    """
    def shuffled_version(dataset: DataSet) -> Iterable:
        """
        Wrap the data_generator in this function.

        Args:
            dataset: just a dataset.

        Returns:
            shuffled version of the data generator.

        """
        new_dataset = shuffle_dataset(dataset)
        return data_generator(new_dataset)
    return shuffled_version


def pre_balance(data_generator: Callable) -> Callable:
    """
    Decorate a data generator function with the balance function.

    Args:
        data_generator: a function that takes a dataset and yield samples.

    Returns:
        balance the dataset before the data_generator is applied.

    """
    def balanced_version(dataset: DataSet) -> Iterable:
        """
        Wrap the data_generator in this function.

        Args:
            dataset: just a dataset.

        Returns:
            balanced version of the data generator.

        """
        new_dataset = balance_dataset(dataset)
        return data_generator(new_dataset)
    return balanced_version


def pre_split(sections: Sequence) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clip function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            split the dataset before the data_generator is applied.

        """
        def split_version(dataset: DataSet) -> Iterable:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                split version of the data generator.

            """
            new_dataset = split_dataset(dataset, sections)

            @extend_to_split_datasets
            def gen(ds):
                return data_generator(ds)

            return gen(new_dataset)
        return split_version
    return decorator


def pre_clip(max_spl: int) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clip function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clip the dataset before the data_generator is applied.

        """
        def clipped_version(dataset: DataSet) -> Iterable:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                clipped version of the data generator.

            """
            new_dataset = clip_dataset(dataset, max_spl)
            return data_generator(new_dataset)
        return clipped_version
    return decorator


def pre_batch(batch_size: int, keep_last: bool = False) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the batch function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            batch the dataset before the data_generator is applied.

        """
        def batched_version(dataset: DataSet) -> Iterable:
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


def pre_clean(dtype: type, rm: Sequence[Any]) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clean function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def cleaned_version(dataset: DataSet) -> Iterable:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            new_dataset = clean_dataset(dataset, dtype, rm)
            return data_generator(new_dataset)
        return cleaned_version
    return decorator


def pre_be_fair(dtype: type, rm: Sequence[Any]) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clean function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def fair_version(dataset: DataSet) -> Iterable:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            new_dataset = fair_dataset(dataset, dtype, rm)
            return data_generator(new_dataset)
        return fair_version
    return decorator


def post_shuffle(dataset_creator: Callable) -> Callable:
    """
    Decorate a dataset creator function with the shuffle function.

    Args:
        dataset_creator: a function that takes any arguments and returns a dataset.

    Returns:
        shuffle the dataset after creation.

    """
    def shuffled_version(*args, **kwargs) -> RefDataSet:
        """
        Wrap the dataset creator in this function.

        Returns:
            shuffled version of the dataset creator.

        """
        new_dataset = dataset_creator(*args, **kwargs)
        return shuffle_dataset(new_dataset)
    return shuffled_version


def post_balance(dataset_creator: Callable) -> Callable:
    """
    Decorate a dataset creator function with the balance function.

    Args:
        dataset_creator: a function that takes any arguments and returns a dataset.

    Returns:
        balance the dataset after creation.

    """
    def balanced_version(*args, **kwargs) -> RefDataSet:
        """
        Wrap the dataset_creator in this function.

        Returns:
            balanced version of the dataset creator.

        """
        new_dataset = dataset_creator(*args, **kwargs)
        return balance_dataset(new_dataset)
    return balanced_version


def post_split(sections: Sequence) -> Callable:
    """Parameterize the decorator."""
    def decorator(dataset_creator: Callable) -> Callable:
        """
        Decorate a dataset creator function with the clip function.

        Args:
            dataset_creator: a function that takes any arguments and returns a dataset.

        Returns:
            split the dataset before the data_generator is applied.

        """
        def split_version(*args, **kwargs) -> SplitDataSet:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                split version of the data generator.

            """
            new_dataset = dataset_creator(*args, **kwargs)
            return split_dataset(new_dataset, sections)
        return split_version
    return decorator


def post_clip(max_spl: int) -> Callable:
    """Parameterize the decorator."""
    def decorator(dataset_creator: Callable) -> Callable:
        """
        Decorate a dataset creator function with the clip function.

        Args:
            dataset_creator: a function that takes any arguments and returns a dataset.

        Returns:
            clip the dataset before the data_generator is applied.

        """
        def clipped_version(*args, **kwargs) -> RefDataSet:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                clipped version of the data generator.

            """
            new_dataset = dataset_creator(*args, **kwargs)
            return clip_dataset(new_dataset, max_spl)
        return clipped_version
    return decorator


def post_clean(dtype: type, rm: Sequence[Any]) -> Callable:
    """Parameterize the decorator."""
    def decorator(dataset_creator: Callable) -> Callable:
        """
        Decorate a dataset creator function with the clean function.

        Args:
            dataset_creator: a function that takes any arguments and returns a dataset.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def cleaned_version(*args, **kwargs) -> RefDataSet:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            new_dataset = dataset_creator(*args, **kwargs)
            return clean_dataset(new_dataset, dtype, rm)
        return cleaned_version
    return decorator


def post_be_fair(dtype: type, rm: Sequence[Any]) -> Callable:
    """Parameterize the decorator."""
    def decorator(dataset_creator: Callable) -> Callable:
        """
        Decorate a dataset creator function with the fair function.

        Args:
            dataset_creator: a function that takes any arguments and returns a dataset.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def fair_version(*args, **kwargs) -> RefDataSet:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            new_dataset = dataset_creator(*args, **kwargs)
            return fair_dataset(new_dataset, dtype, rm)
        return fair_version
    return decorator


def query_slide(
    slides: Dict[str, openslide.OpenSlide],
    patch_size: int
) -> Callable:
    """Parameterize the decorator."""
    def decorator(data_generator: Callable) -> Callable:
        """
        Decorate a data generator function with the clean function.

        Args:
            data_generator: a function that takes a dataset and yield samples.

        Returns:
            clean the dataset before the data_generator is applied.

        """
        def query_version(dataset: DataSet) -> Generator:
            """
            Wrap the data_generator in this function.

            Args:
                dataset: just a dataset.

            Returns:
                cleaned version of the data generator.

            """
            for x, y in data_generator(dataset):
                yield fast_slide_query(slides, x, patch_size), y
        return query_version
    return decorator
