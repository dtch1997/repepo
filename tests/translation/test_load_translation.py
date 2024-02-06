from repepo.translation.load_translation import load_translation, TS


def test_load_translation() -> None:
    assert load_translation(TS.caa_choices) == "Choices:"
    assert load_translation(TS.caa_choices, "fr") == "Choix :"
