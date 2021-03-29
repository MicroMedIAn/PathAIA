from pathaia.patches import izi_filters, filter_image, slide_rois, UnknownFilterError
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

    image = numpy.zeros((64, 64, 3), dtype=numpy.uint8)

    assert filter_image(image, [always_true])
    assert not filter_image(image, [always_false])
    assert filter_image(image, ["true"])
    assert not filter_image(image, ["false"])
    assert not filter_image(image, ["false", always_true])
    assert not filter_image(image, ["true", always_false])
    assert filter_image(image, ["true", always_true])
    raises(UnknownFilterError, filter_image, image, ["this_filter_does_not_exist"])
    raises(UnknownFilterError, filter_image, image, [0])


def test_slide_rois_no_ancestors():
    slide = FakeSlide(name="fake_slide", staining="H&E", extension=".mrxs")
    level = 1
    psize = 224
    interval = {"x": 224, "y": 224}
    patchinfo, image = next(slide_rois(slide, level, psize, interval))
    expected = {
        "id": "#1",
        "x": 0,
        "y": 0,
        "level": 1,
        "dx": 448,
        "dy": 448,
        "parent": "None",
    }
    for k, v in expected.items():
        assert v == patchinfo[k]
