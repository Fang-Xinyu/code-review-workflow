from solution import add


def test_positive():
    assert add(1, 2) == 3


def test_negative():
    assert add(-1, -2) == -3


def test_zero():
    assert add(0, 0) == 0
