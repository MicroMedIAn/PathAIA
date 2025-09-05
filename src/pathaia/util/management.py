"""
Helpful function to extract and organize data.

It takes advantage of the common structure of pathaia projects to enable
datasets creation and experiment monitoring/evaluation.
"""

import pandas as pd
import os
import warnings
from typing import Sequence, Tuple, Iterator, List
from .types import Patch, PathLike
from glob import glob
import numpy as np
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from ..datasets.data import get_tf_dataset
from tqdm import tqdm


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class LevelNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class EmptyProjectError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class SlideNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class PatchesNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class UnknownColumnError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


def get_patch_csv_from_patch_folder(patch_folder: str) -> str:
    """
    Give csv of patches given the slide patch folder.

    Check existence of the path and return absolute path of the csv.

    Args:
        patch_folder: absolute path to a pathaia slide folder.

    Returns:
        Absolute path of csv patch file.

    """
    if os.path.isdir(patch_folder):
        patch_file = os.path.join(patch_folder, "patches.csv")
        if os.path.exists(patch_file):
            return patch_file
        raise PatchesNotFoundError(
            "Could not find extracted patches for the slide: {}".format(
                patch_folder
            )
        )
    raise SlideNotFoundError(
        "Could not find a patch folder at: {}!!!".format(patch_folder)
    )


def get_patch_folders_in_project(project_folder: str) -> Iterator[PathLike]:
    """
    Give pathaia slide folders from a pathaia project folder (direct subfolders).

    Check existence of the project and yield slide folders inside.

    Args:
        project_folder: absolute path to a pathaia project folder.
        exclude: a list of str to exclude from subfolders of the project.
    Yields:
        Absolute path to folder containing patches csv files.

    """
    for folder in glob(os.path.join(project_folder, '*')):
        patch_file = os.path.join(folder, "patches.csv")
        if os.path.exists(patch_file):
            yield folder
        else:
            for f in get_patch_folders_in_project(folder):
                yield f


def get_slide_file(
    slide_folder: str, project_folder: str, patch_folder: str,
    extensions: List[str] = ['.mrxs', '.svs']
) -> str:
    """
    Give the absolute path to a slide file.

    Get the slide absolute path if slide name and slide folder are provided.

    Args:
        slide_folder: absolute path to a folder of WSIs.
        project_folder: absolute path to a pathaia folder.
        patch_folder: absolute path to a folder containing a 'patches.csv'.
    Returns:
        Absolute path of the WSI.

    """
    if not os.path.isdir(slide_folder):
        raise SlideNotFoundError(
            "Could not find a slide folder at: {}!!!".format(slide_folder)
        )
    for ext in extensions:
        slide = patch_folder.replace(project_folder, slide_folder) + ext
        if os.path.exists(slide):
            return slide
    raise SlideNotFoundError(
        "Could not find an {} slide file for: {} in {}!!!".format(
            ext, slide, slide_folder
        )
    )


def read_patch_file(
    patch_file: str, slide_path: str, column: str = None, level: int = None
) -> Iterator[Tuple[dict, str]]:
    """
    Read a patch file.

    Read lines of the patch csv looking for 'column' label.

    Args:
        patch_file: absolute path to a csv patch file.
        level: pyramid level to query patches in the csv.
        slide_path: absolute path to a slide file.
        column: header of the column to use to label individual patches.

    Yields:
        Position and label of patches (x, y, label).

    """
    df = pd.read_csv(patch_file)
    if level is not None:
        df = df[df["level"] == level]
    if column not in df:
        for _, row in df.iterrows():
            yield {
                "x": row["x"],
                "y": row["y"],
                "dx": row["dx"],
                "dy": row["dy"],
                "id": row["id"],
                "level": row["level"],
                "slide_path": slide_path,
                "slide": slide_path,
                "slide_name": os.path.basename(slide_path)
            }, None
    else:
        for _, row in df.iterrows():
            yield {
                "x": row["x"],
                "y": row["y"],
                "dx": row["dx"],
                "dy": row["dy"],
                "id": row["id"],
                "level": row["level"],
                "slide_path": slide_path,
                "slide": slide_path,
                "slide_name": os.path.basename(slide_path)
            }, row[column]


def write_slide_predictions(
    slide_predictions: Iterator[Patch], slide_csv: str, column: str
):
    """
    Write slide predictions in a pathaia slide csv.

    Args:
        slide_predictions: iterator on patch dicts.
        slide_csv: absolute path to a pathaia slide csv.
        column: name of the prediction column to append in csv.

    """
    patch_df = pd.read_csv(slide_csv, sep=None, engine="python")
    patch_df = patch_df.set_index("id")
    for patch in slide_predictions:
        idx = patch["id"]
        pred = patch[column]
        patch_df.loc[idx, column] = pred
    patch_df.to_csv(slide_csv, index=False)


def descriptors_to_csv(
    descriptors: List[Tuple], filename: str, patch_list: List[Patch]
):
    """
    Write patch embeddings into a csv file.

    Args:
        descriptors: list of
        filename:

    """
    columns = ['id', 'level', 'x', 'y']
    descriptors = np.asarray(descriptors)
    for i in range(descriptors.shape[1]):
        columns.append(f'{i}')
    descriptor_df = pd.DataFrame([], columns=columns)
    for x in range(len(patch_list)):
        data = {'id': patch_list[x]['id'],
                'level': patch_list[x]['level'],
                'x': patch_list[x]['x'],
                'y': patch_list[x]['y']}
        for i in range(descriptors.shape[1]):
            data[f'{i}'] = descriptors[x, i]
        descriptor_df = descriptor_df.append(data, ignore_index=True)
    descriptor_df.to_csv(filename, index=False)


class PathaiaHandler(object):
    """
    A class to handle simple patch datasets.

    It usually computes the input of tf datasets proposed in pathaia.data.

    Args:
        project_folder: absolute path to a pathaia project.
        slide_folder: absolute path to a slide folder.

    """

    def __init__(self, project_folder: str, slide_folder: str):
        """Init PathaiaHandler."""
        self.slide_folder = slide_folder
        self.project_folder = project_folder

    def _iter_slides(self) -> Iterator[Tuple[str, str]]:
        """Yield slide folders with associated 'patches.csv'."""
        for folder in get_patch_folders_in_project(self.project_folder):
            try:
                slide_path = get_slide_file(
                    self.slide_folder, self.project_folder, folder
                )
                patch_file = get_patch_csv_from_patch_folder(folder)
            except (
                PatchesNotFoundError, UnknownColumnError, SlideNotFoundError
            ) as e:
                warnings.warn(str(e))
            yield slide_path, patch_file

    def random_split(
        self, ratio: float = 0.3
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Split whole slide dataset into training/validation.

        Args:
            ratio: ratio of slides to keep for validation.
        Returns:
            Training and validation datasets.

        """
        slides = []
        for slide in self._iter_slides():
            slides.append(slide)
        np.random.shuffle(slides)
        validation = slides[0:int(ratio * len(slides))]
        training = slides[int(ratio * len(slides))::]
        return training, validation

    def list_patches(
        self, level: int, dim: Tuple[int, int],
        column: str = None, slides: Iterator = None
    ) -> Tuple[List[Patch], List[str]]:
        """
        Create labeled patch dataset.

        Args:
            level: pyramid level to extract patches in csv.
            dim: dimensions of the patches in pixels.
            label: column header in csv to use as a category.
        Returns:
            List of patch dicts and list of labels.

        """
        patch_list = []
        labels = []
        slide_list = self._iter_slides()
        if slides is not None:
            slide_list = slides
        for slide_path, patch_file in slide_list:
            try:
                # read patch file and get the right level
                for patch, label in read_patch_file(
                    patch_file, slide_path, column, level
                ):
                    patch_list.append(patch)
                    labels.append(label)
            except (
                PatchesNotFoundError, UnknownColumnError, SlideNotFoundError
            ) as e:
                warnings.warn(str(e))
        return patch_list, labels

    def extract_features(
        self,
        model_name: str = 'ResNet50',
        slides: Iterator = None,
        patch_size: int = 224,
        level: int = None,
        layer: str = '',
        batch_size: int = 128
    ):
        """Extract features from patches with a model from keras applications."""
        models = {
            'ResNet50': {
                'model': resnet50.ResNet50,
                'module': resnet50
            }
        }
        preproc = models[model_name]['module'].preprocess_input
        ModelClass = models[model_name]['model']
        model = ModelClass(weights='imagenet', include_top=False,
                           pooling='avg',
                           input_shape=(patch_size, patch_size, 3))
        if not layer == '':
            layer_model = Model(inputs=model.input,
                                outputs=model.get_layer(layer).output)
            model = Sequential()
            model.add(layer_model)
            model.add(GlobalAveragePooling2D())
        slide_list = self._iter_slides()
        if slides is not None:
            slide_list = slides
        for slide_path, patch_file in tqdm(slide_list):
            try:
                patch_list = []
                label_list = []
                # read patch file and get the right level
                for patch, _ in read_patch_file(patch_file, slide_path,
                                                level=level):
                    patch_list.append(patch)
                    label_list.append(0)
            except (
                PatchesNotFoundError, UnknownColumnError, SlideNotFoundError
            ) as e:
                warnings.warn(str(e))
            if len(patch_list) == 0:
                # Raise error here
                print(f'No patches for slide {slide_path}')
                continue
            patch_set = get_tf_dataset(patch_list, label_list, preproc,
                                       batch_size=batch_size,
                                       patch_size=patch_size,
                                       training=False)
            descriptors = model.predict(patch_set)
            descriptor_csv = os.path.join(
                os.path.dirname(patch_file), f'features_{model_name}.csv'
            )
            descriptors_to_csv(descriptors, descriptor_csv, patch_list)
