from pathaia.util.basic import ifnone


def test_ifnone():
    a = None
    b = 1
    assert ifnone(a, b) == b
    a = 2
    assert ifnone(a, b) == a
