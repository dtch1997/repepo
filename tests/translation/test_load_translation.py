from repepo.translation.load_translation import load_translation
from repepo.translation.translation_strings import TranslationString as T


def test_load_translation() -> None:
    assert load_translation(T.caa_choices, "en") == "Choices:"
    assert load_translation(T.caa_choices, "fr") == "Choix :"
