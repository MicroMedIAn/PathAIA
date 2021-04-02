from pathaia.util.types import Coord


def test_coord():
    x1, y1 = coord1 = Coord(4, 6)
    x2, y2 = coord2 = Coord(2, 8)
    a = 3
    b = 1.6

    assert Coord(a) == Coord(a, a)
    assert Coord(a)[0] == a
    assert Coord(b)[1] == b

    assert -coord1 == Coord(-4, -6)
    assert coord1 + coord2 == coord2 + coord1 == Coord(x1 + x2, y1 + y2)
    assert coord1 * coord2 == coord2 * coord1 == Coord(x1 * x2, y1 * y2)
    assert coord1 / coord2 == coord1 // coord2 == Coord(x1 // x2, y1 // y2)
    assert coord2 / coord1 == coord2 // coord1 == Coord(x2 // x1, y2 // y1)
    assert coord1 - coord2 == -coord2 + coord1 == Coord(x1 - x2, y1 - y2)
    assert coord2 - coord1 == -coord1 + coord2 == Coord(x2 - x1, y2 - y1)

    assert coord1 + a == a + coord1 == Coord(a + x1, a + y1)
    assert coord1 * a == a * coord1 == Coord(a * x1, a * y1)
    assert coord1 / a == coord1 // a == Coord(x1 // a, y1 // a)
    assert a / coord1 == a // coord1 == Coord(a // x1, a // y1)
    assert coord1 - a == -a + coord1 == Coord(x1 - a, y1 - a)
    assert a - coord1 == -coord1 + a == Coord(a - x1, a - y1)

    assert coord1 + b == b + coord1 == Coord(int(x1 + b), int(y1 + b))
    assert coord1 * b == b * coord1 == Coord(int(x1 * b), int(y1 * b))
    assert coord1 / b == coord1 // b == Coord(int(x1 / b), int(y1 / b))
    assert b / coord1 == b // coord1 == Coord(int(b / x1), int(b / y1))
    assert coord1 - b == -b + coord1 == Coord(int(x1 - b), int(y1 - b))
    assert b - coord1 == -coord1 + b == Coord(int(b - x1), int(b - y1))
