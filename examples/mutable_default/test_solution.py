from solution import append_item


def test_default_is_independent():
    assert append_item(1) == [1]
    assert append_item(2) == [2]


def test_explicit_list_keeps_identity():
    values = [3]
    assert append_item(4, values) is values
    assert values == [3, 4]
