"""A module to handle data generation for deep neural networks.

It uses the tf.data.Dataset object to enable parallel computing of batches.
"""
import numpy as np
import openslide
import tensorflow as tf
from typing import Sequence, Callable, Iterator, Any, Tuple, Optional
from ..util.types import Patch, NDByteImage


def slide_query(patch: Patch) -> NDByteImage:
    """
    Query patch image in slide.

    Get patch image given position, level and dimensions.

    Args:
        patch: the patch to query.

    Returns:
        Numpy array rgb image of the patch.

    """
    slide = openslide.OpenSlide(patch["slide"])
    pil_img = slide.read_region(
        (patch["x"], patch["y"]), patch["level"], patch["dimensions"]
    )
    return np.array(pil_img)[:, :, 0:3]


def generator_fn(
    patch_list: Sequence[Patch], label_list: Sequence[Any], preproc: Callable
) -> Iterator[Tuple[Patch, Any]]:
    """
    Provide a generator for tf.data.Dataset.

    Create a scope with required arguments, but produce a arg-less iterator.

    Args:
        patch_list: patch list to query.
        label_list: label of patches.
        preproc: a preprocessing function for images.
    Returns:
        A generator of tuples (patch, label).

    """

    def generator():
        for patch, y in zip(patch_list, label_list):
            x = slide_query(patch)
            yield preproc(x), y

    return generator


def get_tf_dataset(
    patch_list: Sequence[Patch],
    label_list: Any,
    preproc: Callable,
    batch_size: int,
    patch_size: int,
    prefetch: Optional[int] = None,
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
    gen = generator_fn(patch_list, label_list, preproc)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(np.float32, np.int32),
        output_shapes=((patch_size, patch_size, 3), label_list[0].shape),
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # prefetch
    # <=> while fitting batch b, prepare b+k in parallel
    if prefetch is None:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(prefetch)
    return dataset
