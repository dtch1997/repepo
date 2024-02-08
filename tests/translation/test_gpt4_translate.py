from unittest.mock import MagicMock

from pytest_mock import MockerFixture
from repepo.translation.gpt4_translate import (
    gpt4_style_translate,
    gpt4_language_translate,
)


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
