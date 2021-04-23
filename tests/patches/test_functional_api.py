from pathaia.patches import (
    izi_filters,
    slide_filters,
    filter_image,
    apply_slide_filters,
    slide_rois,
    UnknownFilterError,
    Patch,
)
from tests.helpers import FakeSlide
import numpy
from pytest import raises


def test_filter_image():
    def always_true(x):
        return True

    def always_false(x):
        return False

    izi_filters["true"] = always_true
    izi_filters["false"] = always_false

    images = [
        numpy.zeros((64, 64, 3), dtype=dtype)
        for dtype in (numpy.uint8, numpy.float32, numpy.float64)
    ]
    for image in images:
        assert filter_image(image, [always_true])
        assert not filter_image(image, [always_false])
        assert filter_image(image, ["true"])
        assert not filter_image(image, ["false"])
        assert not filter_image(image, ["false", always_true])
        assert not filter_image(image, ["true", always_false])
        assert filter_image(image, ["true", always_true])
        raises(UnknownFilterError, filter_image, image, ["this_filter_does_not_exist"])
        raises(UnknownFilterError, filter_image, image, [0])


def test_apply_slide_filters():
    def always_true(x):
        return numpy.ones(x.shape[:2], dtype=bool)

    def always_false(x):
        return numpy.zeros(x.shape[:2], dtype=bool)

    slide_filters["true"] = always_true
    slide_filters["false"] = always_false

    images = [
        numpy.zeros((64, 64, 3), dtype=dtype)
        for dtype in (numpy.uint8, numpy.float32, numpy.float64)
    ]
    for image in images:
        assert apply_slide_filters(image, [always_true]).all()
        assert not apply_slide_filters(image, [always_false]).any()
        assert apply_slide_filters(image, ["true"]).all()
        assert not apply_slide_filters(image, ["false"]).any()
        assert not apply_slide_filters(image, ["false", always_true]).any()
        assert not apply_slide_filters(image, ["true", always_false]).any()
        assert apply_slide_filters(image, ["true", always_true]).all()
        raises(UnknownFilterError, filter_image, image, ["this_filter_does_not_exist"])


def test_slide_rois():
    slide = FakeSlide(name="fake_slide", staining="H&E", extension=".mrxs")
    level = 1
    psize = 224
    interval = (0, 0)
    dsr = slide.level_downsamples[level]
    patch, image = next(slide_rois(slide, level, psize, interval))
    expected = Patch(
        id="#1",
        slidename=slide._filename,
        position=(0, 0),
        level=1,
        size=(psize, psize),
        size_0=(int(psize * dsr), int(psize * dsr)),
    )
    assert patch == expected

    ancestors = [expected]
    level -= 1
    dsr = slide.level_downsamples[level]
    patch, image = next(slide_rois(slide, level, psize, interval, ancestors=ancestors))
    expected = Patch(
        id="#1#1",
        slidename=slide._filename,
        position=(0, 0),
        level=0,
        size=(psize, psize),
        size_0=(int(psize * dsr), int(psize * dsr)),
        parent=ancestors[0],
    )
    assert patch == expected
