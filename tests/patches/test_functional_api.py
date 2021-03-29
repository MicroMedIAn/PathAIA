import unittest
from pathaia.patches import slide_rois
from ..helpers import FakeSlide


class TestPatchify(unittest.TestCase):
    """
    """

    slide = FakeSlide(name="fake_slide", staining="H&E", extension=".mrxs")

    def test_slide_rois(self):
        level = 1
        psize = 224
        interval = {"x": 224, "y": 224}
        patchinfo, image = next(slide_rois(self.slide, level, psize, interval))
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
            self.assertEqual(v, patchinfo[k])


if __name__ == "__main__":
    unittest.main()
