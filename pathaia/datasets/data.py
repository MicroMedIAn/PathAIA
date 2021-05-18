"""A module to handle data generation for deep neural networks.

It uses the tf.data.Dataset object to enable parallel computing of batches.
"""
import numpy as np
import openslide
import tensorflow as tf
from typing import Sequence, Callable, Iterator, Any, Tuple, Optional, Dict, Union
from ..util.types import Patch, NDByteImage


def slide_query(patch: Patch, patch_size: int) -> NDByteImage:
    """
    Query patch image in slide.

    Get patch image given position, level and dimensions.

    Args:
        patch: the patch to query.
        patch_size: size of side of the patch in pixels.

    Returns:
        Numpy array rgb image of the patch.

    """
    slide = openslide.OpenSlide(patch["slide"])
    pil_img = slide.read_region(
        (patch["x"], patch["y"]), patch["level"], (patch_size, patch_size)
    )
    return np.array(pil_img)[:, :, 0:3]


def fast_slide_query(
    slides: Dict[str, openslide.OpenSlide],
    patch: Patch,
    patch_size: int
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
    slide = slides[patch["slide"]]
    pil_img = slide.read_region(
        (patch["x"], patch["y"]), patch["level"], (patch_size, patch_size)
    )
    return np.array(pil_img)[:, :, 0:3]


def generator_fn(
    patch_list: Sequence[Patch],
    label_list: Sequence[Any],
    patch_size: int,
    preproc: Callable
) -> Iterator[Tuple[Patch, Any]]:
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
            x = slide_query(patch, patch_size)
            yield preproc(x), y

    return generator


def get_tf_dataset(
    patch_list: Sequence[Patch],
    label_list: Any,
    preproc: Callable,
    batch_size: int,
    patch_size: int,
    prefetch: Optional[int] = None,
    training: Optional[bool] = True,
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

    Returns:
        tf.data.Dataset: a proper tensorflow dataset to fit on.

    """
    gen = generator_fn(patch_list, label_list, patch_size, preproc)
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
