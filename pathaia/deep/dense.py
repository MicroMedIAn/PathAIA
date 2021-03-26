# coding: utf8
"""Useful functions and classes to store vocabularies learned on images.

This module gather functions and classes to handle vocabulary learning and
inference on image datasets.
"""
from sklearn.cluster import MiniBatchKMeans, KMeans
from ..util.images import images_in_folder, sample_img_sep_channels, sample_img
from ..util.paths import dataset2folders
from ..util.types import NDImageBatch, NDBoolMaskBatch, PathLike
from matplotlib import pyplot as plt
import numpy
import os
import json
from typing import Set, Optional, Union, Any, List, Sequence
from nptyping import NDArray


class Vocabulary(object):
    """An object to store an image vocabulary.

    This class defines parameters and learning/inference behaviour of an
    image vocabulary.

    """

    def __init__(
        self,
        context: int,
        size: int,
        level: int,
        img_per_slide: int,
        ptc_per_img: int,
        dataset_len: int,
        n_channels: int,
    ):
        self._context = context
        self._n_words = size
        self._level = level
        self._img_per_slide = img_per_slide
        self._ptc_per_img = ptc_per_img
        self._dataset_len = dataset_len
        self._n_channels = n_channels
        self._clf = MiniBatchKMeans(n_clusters=size, reassignment_ratio=0.0)
        self._training_slides = set()
        self._centroids = dict()
        self._annotations = dict()

    @property
    def training_slides(self) -> Set[str]:
        """
        The set of slides used to obtain the centroids of the model.

        """
        return self._training_slides

    @property
    def context(self) -> int:
        """
        The size of a patch-word.

        """
        return self._context

    @property
    def level(self) -> int:
        """
        The pyramid level (magnification) at which extract patches in slide.

        """
        return self._level

    @property
    def n_words(self) -> int:
        """
        The number of words, i.e. centroids.

        """
        return self._n_words

    @property
    def img_per_slide(self) -> int:
        """
        The number of 'big' images to take into account
        to fit on a single slide.

        """
        return self._img_per_slide

    @property
    def ptc_per_img(self) -> int:
        """
        The number of 'small' images to extract from a 'big'
        image in a slide.

        """
        return self._ptc_per_img

    @property
    def n_channels(self) -> int:
        """
        The number of channels in the images we fit on.

        """
        return self._n_channels

    def fit_on_imbatch(self, batch: NDImageBatch, mask_batch: NDBoolMaskBatch = None):
        """Partial fit of the k-means model used as Vocabulary.

        Fit the k-means clf on a new batch of data.

        Args:
            batch: batch of big images to fit on.

        """
        ptcs = []
        if mask_batch is None:
            for img in batch:
                ptcs += sample_img(img, self._context, self._ptc_per_img)
        else:
            for img, msk in zip(batch, mask_batch):
                ptcs += sample_img(img, self._context, self._ptc_per_img, msk)
        self._clf.partial_fit(ptcs)

    def predict(
        self, batch: NDArray[(Any, Any), float], fuzzy: bool = False
    ) -> Union[NDArray[(Any,), int], NDArray[(Any, Any), int]]:
        """Predict with the k-means model.

        Predict with the k-means clf on a new batch of data.

        Args:
            batch: batch of flattened patches to predict.

        Returns:
            Predictions on batch of data.

        """
        if fuzzy:
            return self._clf.transform(batch)
        return self._clf.predict(batch)

    def fit_on_slide(self, slide_ptc_folder: PathLike):
        """Fit Vocabulary on a single slide batch.

        Args:
            slide_ptc_folder: path to an image folder.

        """
        batch = []
        for img in images_in_folder(
            slide_ptc_folder, randomize=True, datalim=self._img_per_slide
        ):
            batch.append(img)
        self.fit_on_imbatch(batch)

    def fit_on_dataset(
        self, dataset_folder: PathLike, verbose: int, outfolder: Optional[PathLike] = None
    ):
        """Fit vocabulary on several batches taken from several slides.

        Args:
            dataset_folder: path to a pathaia dataset folder.
            level: pyramid level at which extract patches.
            verbose: degree of console output while fitting.

        """
        slide2folder = dataset2folders(
            dataset_folder,
            self._level,
            randomize=True,
            slide_data_lim=self._dataset_len,
        )
        k = 0
        for slidename, folder in slide2folder.items():
            self._training_slides.add(slidename)
            k += 1
            if verbose > 0:
                print("processing slide {} / {}".format(k, len(slide2folder)))
                print("fitting on slide name: {}".format(slidename))
                print("fitting on level: {}".format(self._level))
            self.fit_on_slide(folder)

        for idx, center in enumerate(self._clf.cluster_centers_):
            self._centroids[idx] = center

        # affichage
        if verbose > 1 and outfolder is not None:
            fig = plt.figure()
            patch_shape = (self._context, self._context, 3)
            for i in range(int(numpy.sqrt(self._n_words))):
                for j in range(int(numpy.sqrt(self._n_words))):
                    image = self._clf.cluster_centers_[
                        i * int(numpy.sqrt(self._n_words)) + j
                    ].reshape(patch_shape)
                    print(
                        "stats of filter {}:\nmin={}, max={}, mean={}, dynamic={}".format(
                            i * int(numpy.sqrt(self._n_words)) + j,
                            image.min(),
                            image.max(),
                            image.mean(),
                            len(numpy.unique(image)),
                        )
                    )
                    ax = fig.add_subplot(
                        int(numpy.sqrt(self._n_words)),
                        int(numpy.sqrt(self._n_words)),
                        i * int(numpy.sqrt(self._n_words)) + j + 1,
                    )
                    # Turn off tick labels
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    image = numpy.around(image)
                    image = image.astype(numpy.uint8)
                    ax.imshow(image)

            plt.savefig(
                os.path.join(
                    outfolder, "vocabulary_{}_{}.png".format(self._context, self._level)
                )
            )

    def to_json(self, filepath: PathLike):
        """Save vocabulary parameters to a json file.

        Put parameters to a dictionary and save it into a json file.

        Args:
            filepath: path to a json file to write.

        """
        output_dict = dict()
        output_dict["context"] = self._context
        output_dict["n_words"] = self._n_words
        output_dict["level"] = self._level
        output_dict["img_per_slide"] = self._img_per_slide
        output_dict["ptc_per_img"] = self._ptc_per_img
        output_dict["dataset_len"] = self._dataset_len
        output_dict["training_slides"] = list(self._training_slides)
        output_dict["n_channels"] = self._n_channels
        output_dict["centroids"] = {k: list(v) for k, v in self._centroids.items()}

        json_txt = json.dumps(output_dict)
        with open(filepath, "w") as outputjson:
            outputjson.write(json_txt)

    def from_json(self, filepath: PathLike):
        """Load vocabulary from json.

        Read json as a dictionary and put params in object's attributes.

        Args:
            filepath: path to a json file to read.

        """
        with open(filepath, "r") as inputjson:
            input_dict = json.load(inputjson)

        self._context = input_dict["context"]
        self._n_words = input_dict["n_words"]
        self._level = input_dict["level"]
        self._img_per_slide = input_dict["img_per_slide"]
        self._ptc_per_img = input_dict["ptc_per_img"]
        self._dataset_len = input_dict["dataset_len"]
        self._training_slides = input_dict["training_slides"]
        self._n_channels = input_dict["n_channels"]
        self._centroids = {
            k: numpy.array(v) for k, v in input_dict["centroids"].items()
        }
        self._clf = KMeans()
        centroid_idx_list = sorted(self._centroids.keys())
        centroids = []
        for k in centroid_idx_list:
            centroids.append(self._centroids[k])
        self._clf.cluster_centers_ = numpy.array(centroids)


class SepChannelsVocabulary(object):
    """An object to store an image vocabulary.

    This class defines parameters and learning/inference behaviour of an
    image vocabulary.

    """

    def __init__(
        self,
        context: int,
        size: int,
        level: int,
        img_per_slide: int,
        ptc_per_img: int,
        dataset_len: int,
        n_channels: int,
    ):
        self._context = context
        self._n_words = size
        self._level = level
        self._img_per_slide = img_per_slide
        self._ptc_per_img = ptc_per_img
        self._dataset_len = dataset_len
        self._n_channels = n_channels
        self._clf = []
        self._centroids = []
        for c in range(self._n_channels):
            self._clf.append(MiniBatchKMeans(n_clusters=size, reassignment_ratio=0.0))
            self._centroids.append(dict())
        self._training_slides = set()

    @property
    def training_slides(self) -> Set[str]:
        """
        The set of slides used to obtain the centroids of the model.

        """
        return self._training_slides

    @property
    def context(self) -> int:
        """
        The side of a patch-word.

        """
        return self._context

    @property
    def level(self) -> int:
        """
        The pyramid level (magnification) at which extract patches in slide.

        """
        return self._level

    @property
    def n_words(self) -> int:
        """
        The number of words, i.e. centroids.

        """
        return self._n_words

    @property
    def img_per_slide(self) -> int:
        """
        The number of 'big' images to take into account
        to fit on a single slide.

        """
        return self._img_per_slide

    @property
    def ptc_per_img(self) -> int:
        """
        The number of 'small' images to extract from a 'big'
        image in a slide.

        """
        return self._ptc_per_img

    @property
    def n_channels(self) -> int:
        """
        The number of channels in the images we fit on.

        """
        return self._n_channels

    def fit_on_imbatch(
        self, batch: NDImageBatch, mask_batch: Optional[NDBoolMaskBatch] = None
    ):
        """Partial fit of the k-means model used as Vocabulary.

        Fit the k-means clf on a new batch of data.

        Args:
            batch: batch of big images to fit on.

        """
        ch_ptcs = [[] for c in range(self._n_channels)]
        for img in batch:
            for idx, channel_patches in enumerate(
                sample_img_sep_channels(img, self._context, self._ptc_per_img)
            ):
                ch_ptcs[idx] += channel_patches
        for clf, ptcs in zip(self._clf, ch_ptcs):
            clf.partial_fit(ptcs)

    def predict(
        self, batch: Sequence[NDArray[(Any, Any), float]], fuzzy=False
    ) -> Union[List[NDArray[(Any,), int]], List[NDArray[(Any, Any), int]]]:
        """Predict with the k-means model.

        Predict with the k-means clf on a new batch of data.

        Args:
            batch: flattened patches to predict, one list per channel.

        Returns:
            Predictions on batch of data.

        """
        preds = []
        if fuzzy:
            for clf, ptcs in zip(self._clf, batch):
                preds.append(clf.transform(ptcs))
        else:
            for clf, ptcs in zip(self._clf, batch):
                preds.append(clf.predict(ptcs))
        return preds

    def fit_on_slide(self, slide_ptc_folder: PathLike):
        """Fit Vocabulary on a single slide batch.

        Args:
            slide_ptc_folder: path to an image folder.

        """
        batch = []
        for img in images_in_folder(
            slide_ptc_folder, randomize=True, datalim=self._img_per_slide
        ):
            batch.append(img)
        self.fit_on_imbatch(batch)

    def fit_on_dataset(
        self,
        dataset_folder: PathLike,
        verbose: int,
        outfolder: Optional[PathLike] = None,
    ):
        """Fit vocabulary on several batches taken from several slides.

        Args:
            dataset_folder: path to a pathaia dataset folder.
            level: pyramid level at which extract patches.
            verbose: degree of console output while fitting.

        """
        slide2folder = dataset2folders(
            dataset_folder,
            self._level,
            randomize=True,
            slide_data_lim=self._dataset_len,
        )
        k = 0
        for slidename, folder in slide2folder.items():
            self._training_slides.add(slidename)
            k += 1
            if verbose > 0:
                print("processing slide {} / {}".format(k, len(slide2folder)))
                print("fitting on slide name: {}".format(slidename))
                print("fitting on level: {}".format(self._level))
            self.fit_on_slide(folder)

        for clf_idx, clf in enumerate(self._clf):
            for idx, center in enumerate(clf.cluster_centers_):
                self._centroids[clf_idx][idx] = center

        # affichage
        if verbose > 1 and outfolder is not None:
            for c in range(self._n_channels):
                fig = plt.figure()
                patch_shape = (self._context, self._context)
                for i in range(int(numpy.sqrt(self._n_words))):
                    for j in range(int(numpy.sqrt(self._n_words))):
                        image = (
                            self._clf[c]
                            .cluster_centers_[i * int(numpy.sqrt(self._n_words)) + j]
                            .reshape(patch_shape)
                        )
                        print(
                            "stats of filter {}:\nmin={}, max={}, mean={}, dynamic={}".format(
                                i * int(numpy.sqrt(self._n_words)) + j,
                                image.min(),
                                image.max(),
                                image.mean(),
                                len(numpy.unique(image)),
                            )
                        )
                        ax = fig.add_subplot(
                            int(numpy.sqrt(self._n_words)),
                            int(numpy.sqrt(self._n_words)),
                            i * int(numpy.sqrt(self._n_words)) + j + 1,
                        )
                        # Turn off tick labels
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])
                        image = numpy.around(image)
                        image = image.astype(numpy.uint8)
                        ax.imshow(image)

                plt.savefig(
                    os.path.join(
                        outfolder,
                        "vocabulary_{}_{}_channel_{}.png".format(
                            self._context, self._level, c
                        ),
                    )
                )

    def to_json(self, filepath: PathLike):
        """Save vocabulary parameters to a json file.

        Put parameters to a dictionary and save it into a json file.

        Args:
            filepath: path to a json file to write.

        """
        output_dict = dict()
        output_dict["context"] = self._context
        output_dict["n_words"] = self._n_words
        output_dict["level"] = self._level
        output_dict["img_per_slide"] = self._img_per_slide
        output_dict["ptc_per_img"] = self._ptc_per_img
        output_dict["dataset_len"] = self._dataset_len
        output_dict["training_slides"] = list(self._training_slides)
        output_dict["n_channels"] = self._n_channels
        output_dict["centroids"] = []
        for c in range(self._n_channels):
            output_dict["centroids"].append(
                {k: list(v) for k, v in self._centroids[c].items()}
            )

        json_txt = json.dumps(output_dict)
        with open(filepath, "w") as outputjson:
            outputjson.write(json_txt)

    def from_json(self, filepath: PathLike):
        """Load vocabulary from json.

        Read json as a dictionary and put params in object's attributes.

        Args:
            filepath: path to a json file to read.

        """
        with open(filepath, "r") as inputjson:
            input_dict = json.load(inputjson)

        self._context = input_dict["context"]
        self._n_words = input_dict["n_words"]
        self._level = input_dict["level"]
        self._img_per_slide = input_dict["img_per_slide"]
        self._ptc_per_img = input_dict["ptc_per_img"]
        self._dataset_len = input_dict["dataset_len"]
        self._training_slides = input_dict["training_slides"]
        self._n_channels = input_dict["n_channels"]
        self._centroids = []
        self._clf = []
        for c in range(self._n_channels):
            self._centroids.append(
                {k: numpy.array(v) for k, v in input_dict["centroids"][c].items()}
            )
            km = KMeans()
            centroid_idx_list = sorted(self._centroids[c].keys())
            centroids = []
            for k in centroid_idx_list:
                centroids.append(self._centroids[c][k])
            km.cluster_centers_ = numpy.array(centroids)
            self._clf.append(km)
