from pathaia.util.images import get_coords_from_mask, regular_grid
import numpy as np


def test_regular_grid():
    shape = (8, 12)
    interval = 0
    psize = 4
    expected = [(0, 0), (0, 4), (0, 8), (4, 0), (4, 4), (4, 8)]
    assert expected == sorted(list(regular_grid(shape, interval, psize)))

    interval = -1
    expected = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6)]
    assert expected == sorted(list(regular_grid(shape, interval, psize)))

    interval = 1
    expected = [(0, 0), (0, 5)]
    assert expected == sorted(list(regular_grid(shape, interval, psize)))


def test_get_coords_from_mask():
    shape = (8, 12)
    interval = 0
    psize = 4
    mask = np.zeros((3, 2), dtype=bool)
    expected = []
    assert expected == sorted(list(get_coords_from_mask(mask, shape, interval, psize)))

    mask[1, 1] = True
    expected = [(4, 4)]
    assert expected == sorted(list(get_coords_from_mask(mask, shape, interval, psize)))

    mask = np.ones((3, 2), dtype=bool)
    assert sorted(list(regular_grid(shape, interval, psize))) == sorted(
        list(get_coords_from_mask(mask, shape, interval, psize))
    )
