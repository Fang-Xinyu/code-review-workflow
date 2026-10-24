import pytest
from solution import divide


def test_positive():
    assert divide(6, 2) == 3


def test_negative():
    assert divide(-6, 2) == -3


def test_zero():
    with pytest.raises(ValueError):
        divide(1, 0)
