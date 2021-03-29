from tests.data import HEMASK
from PIL import Image
import numpy
from openslide import AbstractSlide


class FakeSlide(AbstractSlide):

    """
    A class to mimic an openslide.OpenSlide object.

    Args:
        staining: type of staining you wanna mimic.
    """

    def __init__(self, name="fake_slide", staining="H&E", extension=".mrxs"):
        """
        """
        self._filename = name + extension
        self.tissue_color = [154, 120, 156, 255]

    @property
    def level_count(self):
        return 11

    @property
    def level_dimensions(self):
        return (
            (83968, 71680),
            (41984, 35840),
            (20992, 17920),
            (10496, 8960),
            (5248, 4480),
            (2624, 2240),
            (1312, 1120),
            (656, 560),
            (328, 280),
            (164, 140),
            (82, 70),
        )

    @property
    def level_downsamples(self):
        return (
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
            256.0,
            512.0,
            1024.0,
        )

    @property
    def properties(self):
        """
        """
        return {}

    @property
    def associated_images(self):
        return {}

    def get_best_level_for_downsample(self, downsample):
        for k in range(1, self.level_count):
            if self.level_downsamples[k] > downsample:
                return k - 1
        return self.level_count

    def read_region(self, location, level, size):
        """
        """
        # un pack request coordinates
        x, y = location
        dx, dy = size
        ds = self.level_downsamples[level]
        tds = self.level_downsamples[-1]
        dj, di = self.level_dimensions[-1]

        X_indices = numpy.zeros((dy, dx), dtype=float)
        X_indices += x
        Y_indices = numpy.zeros((dy, dx), dtype=float)
        Y_indices += y
        # go through columns of X
        for column in range(X_indices.shape[1]):
            X_indices[:, column] += column * ds
        # go through all lines of Y
        for line in range(Y_indices.shape[0]):
            Y_indices[line, :] += line * ds
        # rescale x and y indices
        J_indices = numpy.floor(X_indices / tds).astype(int)
        I_indices = numpy.floor(Y_indices / tds).astype(int)
        # put a flag on out-of-bounds pixels
        J_indices[J_indices >= dj] = -1
        I_indices[I_indices >= di] = -1
        # compute 1D indices
        Indices = I_indices * dj + J_indices
        labels = numpy.zeros_like(Indices)
        for index in numpy.unique(Indices):
            # funny condition to allow out-of-slide regions like openslide ;-)
            if index > 0:
                labels[Indices == index] = HEMASK[index]
        numpy_img = numpy.zeros((dy, dx, 4), dtype=numpy.uint8)
        # out of bounds regions have val == 0 ^^
        numpy_img[labels == 0] = (0, 0, 0, 0)
        numpy_img[labels == 1] = (255, 255, 255, 255)
        numpy_img[labels == 2] = self.tissue_color
        # get pil image
        return Image.fromarray(numpy_img, mode="RGBA")
