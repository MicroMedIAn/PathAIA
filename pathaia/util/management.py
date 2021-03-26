"""
Helpful function to extract and organize data.

It takes advantage of the common structure of pathaia projects to enable
datasets creation and experiment monitoring/evaluation.
"""

import pandas as pd
import os
import warnings
from typing import Sequence, Tuple, Iterator, List
from .types import Patch


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
            "Could not find extracted patches for the slide: {}".format(patch_folder)
        )
    raise SlideNotFoundError(
        "Could not find a patch folder at: {}!!!".format(patch_folder)
    )


def get_patch_folders_in_project(
    project_folder: str, exclude: Sequence[str] = ("annotation",)
) -> Iterator[Tuple[str, str]]:
    """
    Give pathaia slide folders from a pathaia project folder (direct subfolders).

    Check existence of the project and yield slide folders inside.

    Args:
        project_folder: absolute path to a pathaia project folder.
        exclude: a list of str to exclude from subfolders of the project.
    Yields:
        Name of the slide and absolute path to its pathaia folder.

    """
    if not os.path.isdir(project_folder):
        raise EmptyProjectError(
            "Did not find any project at: {}".format(project_folder)
        )
    for name in os.listdir(project_folder):
        keep = True
        for ex in exclude:
            if ex in name:
                keep = False
        if keep:
            patch_folder = os.path.join(project_folder, name)
            if os.path.isdir(patch_folder):
                yield name, patch_folder


def get_slide_file(slide_folder: str, slide_name: str) -> str:
    """
    Give the absolute path to a slide file.

    Get the slide absolute path if slide name and slide folder are provided.

    Args:
        slide_folder: absolute path to a folder of WSIs.
        slide_name: basename of the slide.
    Returns:
        Absolute path of the WSI.

    """
    if not os.path.isdir(slide_folder):
        raise SlideNotFoundError(
            "Could not find a slide folder at: {}!!!".format(slide_folder)
        )
    for name in os.listdir(slide_folder):
        if name.endswith(".mrxs") and not name.startswith("."):
            base, _ = os.path.splitext(name)
            if slide_name == base:
                return os.path.join(slide_folder, name)
    raise SlideNotFoundError(
        "Could not find an mrxs slide file for: {} in {}!!!".format(
            slide_name, slide_folder
        )
    )


def handle_labeled_patches(
    patch_file: str, level: int, column: str
) -> Iterator[Tuple[int, int, str]]:
    """
    Read a patch file.

    Read lines of the patch csv looking for 'column' label.

    Args:
        patch_file: absolute path to a csv patch file.
        level: pyramid level to query patches in the csv.
        column: header of the column to use to label individual patches.

    Yields:
        Position and label of patches (x, y, label).

    """
    df = pd.read_csv(patch_file)
    level_df = df[df["level"] == level]
    if column not in level_df:
        raise UnknownColumnError(
            "Column {} does not exists in {}!!!".format(column, patch_file)
        )
    for _, row in level_df.iterrows():
        yield row["x"], row["y"], row[column]


class PathaiaHandler(object):
    """
    A class to handle simple patch datasets.
    It usually computes the input of tf datasets proposed in pathaia.data.

    Args:
        project_folder: absolute path to a pathaia project.
        slide_folder: absolute path to a slide folder.
    """

    def __init__(self, project_folder: str, slide_folder: str):
        self.slide_folder = slide_folder
        self.project_folder = project_folder

    def list_patches(
        self, level: int, dim: Tuple[int, int], label: str
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
        for name, patch_folder in get_patch_folders_in_project(self.project_folder):
            try:
                slide_path = get_slide_file(self.slide_folder, name)
                patch_file = get_patch_csv_from_patch_folder(patch_folder)
                # read patch file and get the right level
                for x, y, lab in handle_labeled_patches(patch_file, level, label):
                    patch_list.append(
                        {
                            "slide": slide_path,
                            "x": x,
                            "y": y,
                            "level": level,
                            "dimensions": dim,
                        }
                    )
                    labels.append(lab)
            except (PatchesNotFoundError, UnknownColumnError, SlideNotFoundError) as e:
                warnings.warn(str(e))
        return patch_list, labels
