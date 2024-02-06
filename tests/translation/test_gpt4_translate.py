from unittest.mock import MagicMock

from pytest_mock import MockerFixture
from repepo.translation.gpt4_translate import (
    translate_strings_parallel,
    gpt4_style_translate,
    gpt4_language_translate,
)


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


def test_gpt4_style_translate(
    mocker: MockerFixture, get_openai_client_mock: MagicMock
) -> None:
    return_val_mock = mocker.MagicMock()
    get_openai_client_mock().chat.completions.create.return_value = return_val_mock
    return_val_mock.choices.__getitem__().message.content = "Arrr this be a test"

    text = "This is a test."
    style = "pirate"
    translation = gpt4_style_translate(text, style)
    assert translation == "Arrr this be a test"


def test_gpt4_language_translate(
    mocker: MockerFixture, get_openai_client_mock: MagicMock
) -> None:
    return_val_mock = mocker.MagicMock()
    get_openai_client_mock().chat.completions.create.return_value = return_val_mock
    return_val_mock.choices.__getitem__().message.content = "Voila un test."

    text = "This is a test."
    language = "fr"
    translation = gpt4_language_translate(text, language)
    assert translation == "Voila un test."
