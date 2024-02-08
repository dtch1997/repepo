from repepo.translation.translate_strings_parallel import translate_strings_parallel


def test_translate_strings_parallel() -> None:
    def translate_fn(orig):
        return f"TRANS: {orig}"

    strings = {"a", "b", "c", "d", "e", "f", "g"}
    results = translate_strings_parallel(strings, translate_fn, max_workers=4)
    assert results == {
        "a": "TRANS: a",
        "b": "TRANS: b",
        "c": "TRANS: c",
        "d": "TRANS: d",
        "e": "TRANS: e",
        "f": "TRANS: f",
        "g": "TRANS: g",
    }
