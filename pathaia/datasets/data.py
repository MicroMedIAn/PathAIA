"""A module to handle data generation for deep neural networks.

It uses the tf.data.Dataset object to enable parallel computing of batches.
"""
from os import cpu_count
from re import M
from typing import (Sequence, Callable, Iterator, Any, Tuple, Optional, Dict,
                    Union, List, Generator)
from functools import lru_cache, partial
import multiprocessing
import numpy as np
import openslide
import tensorflow as tf
from torch.utils.data import IterableDataset, DataLoader
from ..util.types import Patch, NDByteImage


@lru_cache()
def open_slide(slidename: str) -> openslide.OpenSlide:
    """
    Returns a cached version of an OpenSlide object

    Args:
        slidename: path of the slide
    
    Returns:
        OpenSlide object
    """
    return openslide.OpenSlide(slidename)


def slide_query(
    patch: Patch,
    patch_size: int,
    cache: bool = True,
) -> NDByteImage:
    """
    Query patch image in slide.

    Get patch image given position, level and dimensions.

    Args:
        patch: the patch to query.
        patch_size: size of side of the patch in pixels.

    Returns:
        Numpy array rgb image of the patch.

    """
    if cache:
        slide = open_slide(patch.slidename)
    else:
        slide = openslide.OpenSlide(patch.slidename)
    pil_img = slide.read_region((patch.position.x, patch.position.y),
                                patch.level, (patch_size, patch_size))
    return np.array(pil_img)[:, :, 0:3]


def fast_slide_query(
    slides: Dict[str, openslide.OpenSlide],
    patch: Patch,
    patch_size: int,
) -> NDByteImage:
    """
    Query patch image in slide.

    Get patch image given the slide obj, the position, level and dimensions.

    Args:
        slide: the slide to request the patch.
        patch: the patch to query.
        patch_size: size of side of the patch in pixels.

    Returns:
        Numpy array rgb image of the patch.

    """
    slide = slides[patch.slidename]
    pil_img = slide.read_region((patch.position.x, patch.position.y),
                                patch.level, (patch_size, patch_size))
    return np.array(pil_img)[:, :, 0:3]


def generator_fn(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    patch_size: int,
    cache: bool,
    preproc: Callable,
) -> Callable[[], Generator[Tuple[Any, Any], None, None]]:
    """
    Provide a generator for tf.data.Dataset.

    Create a scope with required arguments, but produce a arg-less iterator.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        patch_size: size of the side of the patches in pixels.
        preproc: a preprocessing function for images.
    Returns:
        A generator of tuples (patch, label).

    """
    def generator():
        for patch, y in zip(patch_list, label_list):
            x = slide_query(patch, patch_size, cache)

            yield preproc(x), y

    return generator


def gen(
    patch: Patch,
    y: Any,
    patch_size: int,
    cache: bool,
) -> Tuple[Any, Any]:
    x = slide_query(patch, patch_size, cache)
    return x, y


def gen_wrapper(args, patch_size: int, cache: bool):
    return gen(*args, patch_size=patch_size, cache=cache)


def generator_fn_multi(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    patch_size: int,
    cache: bool,
    preproc: Callable,
) -> Callable[[], Generator[Tuple[Any, Any], None, None]]:
    """
    Provide a generator with multiprocessing for tf.data.Dataset.

    Create a scope with required arguments, but produce a arg-less iterator.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        patch_size: size of the side of the patches in pixels.
        preproc: a preprocessing function for images.
    Returns:
        A function that returns generator of tuples (patch, label).

    """
    def generator():
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() //
                                  2) as pool:
            s = list()
            for x, y in pool.imap(
                    partial(gen_wrapper, patch_size=patch_size, cache=cache),
                    zip(
                        patch_list,
                        label_list,
                    )):
                yield preproc(x), y

    return generator


def get_tf_dataset(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    preproc: Callable,
    batch_size: int,
    patch_size: int,
    cache: Optional[bool] = True,
    prefetch: Optional[int] = None,
    training: Optional[bool] = True,
    parallel: Optional[bool] = False,
) -> tf.data.Dataset:
    """
    Create tensorflow dataset.

    Create tf.data.Dataset object able to prefetch and batch samples from generator.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        preproc: a preprocessing function for images.
        batch_size: number of samples per batch.
        patch_size: size (pixel) of the side of a square patch.
        prefetch: the size of the prefetching (optional),
        training: whether the dataset will be used for training or not,
        parallel: wheter to use parallelism or not

    Returns:
        tf.data.Dataset: a proper tensorflow dataset to fit on.

    """
    if parallel:
        gen = generator_fn_multi(patch_list=patch_list,
                                 label_list=label_list,
                                 patch_size=patch_size,
                                 cache=cache,
                                 preproc=preproc)
    else:
        gen = generator_fn(patch_list=patch_list,
                           label_list=label_list,
                           patch_size=patch_size,
                           cache=cache,
                           preproc=preproc)
    try:
        shape_label = label_list[0].shape
    except AttributeError:
        shape_label = None
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(np.float32, np.int32),
        output_shapes=((patch_size, patch_size, 3), shape_label),
    )
    if training:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)
    # prefetch
    # <=> while fitting batch b, prepare b+k in parallel
    if prefetch is None:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(prefetch)
    return dataset


def get_pt_dataset(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    preproc: Callable,
    batch_size: int,
    patch_size: int,
    cache: Optional[bool] = True,
    training: Optional[bool] = True,
    parallel: Optional[bool] = False,
) -> DataLoader:
    """
    Create pytorch dataset.

    Create tf.data.Dataset object able to prefetch and batch samples from generator.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        preproc: a preprocessing function for images.
        batch_size: number of samples per batch.
        patch_size: size (pixel) of the side of a square patch.
        training: whether the dataset will be used for training or not,
        parallel: wheter to use parallelism or not

    Returns:
        torch.utils.data.Dataloader: a proper pytorch dataset to fit on.

    """
    dataset = TorchDataset(patch_list=patch_list,
                           label_list=label_list,
                           patch_size=patch_size,
                           cache=cache,
                           preproc=preproc,
                           parallel=parallel)
    if training:
        dataloader = InfiniteDataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        drop_last=True)
        return dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            drop_last=False)
    return dataloader


def get_dataset(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    preproc: Callable,
    batch_size: int,
    patch_size: int,
    backend: str,
    cache: Optional[bool] = True,
    prefetch: Optional[int] = None,
    training: Optional[bool] = True,
    parallel: Optional[bool] = False,
) -> Union[tf.data.Dataset, DataLoader]:
    """
    Create either a pytorch dataset or a tf one.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        preproc: a preprocessing function for images.
        batch_size: number of samples per batch.
        patch_size: size (pixel) of the side of a square patch.
        backend: either pt or tf.
        prefetch: Prefetching parameter (only on tensorflow).
        training: whether the dataset will be used for training or not,
        parallel: wheter to use parallelism or not
    Returns:
        Union[tf.data.Dataset, torch.util.data.DataLoader]
    """
    if backend in ['tf', 'tensorflow']:
        return get_tf_dataset(
            patch_list=patch_list,
            label_list=label_list,
            preproc=preproc,
            batch_size=batch_size,
            patch_size=patch_size,
            cache=cache,
            prefetch=prefetch,
            training=training,
            parallel=parallel,
        )
    elif backend in ['pt', 'pytorch']:
        return get_pt_dataset(
            patch_list=patch_list,
            label_list=label_list,
            preproc=preproc,
            batch_size=batch_size,
            patch_size=patch_size,
            cache=cache,
            training=training,
            parallel=parallel,
        )
    else:
        raise ValueError('backend should be either pt or tf')


class TorchDataset(IterableDataset):
    def __init__(
        self,
        patch_list: Sequence[Patch],
        label_list: Sequence[Any],
        patch_size: int,
        cache: bool,
        preproc: Callable,
        parallel: bool,
    ):
        self.patch_list = patch_list
        self.label_list = label_list
        self.patch_size = patch_size
        self.cache = cache
        self.preproc = preproc
        self.parallel = parallel

        if self.parallel:
            self.gen = generator_fn_multi(patch_list=self.patch_list,
                                          label_list=self.label_list,
                                          patch_size=self.patch_size,
                                          cache=self.cache,
                                          preproc=self.preproc)()
        else:
            self.gen = generator_fn(patch_list=self.patch_list,
                                    label_list=self.label_list,
                                    patch_size=self.patch_size,
                                    cache=self.cache,
                                    preproc=self.preproc)()

    def __iter__(self):
        return iter(self.gen)


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
