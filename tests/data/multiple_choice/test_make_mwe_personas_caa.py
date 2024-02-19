from repepo.data.multiple_choice.make_mwe_personas_caa import (
    convert_mwe_personas_dataset_caa,
)
from repepo.translation.load_translation import TS, load_translation


def test_convert_mwe_personas_dataset_caa_english() -> None:
    samples = [
        {
            "question": "What is your name?",
            "answer_matching_behavior": "My name is John Doe.",
            "answer_not_matching_behavior": "I am a person.",
        },
        {
            "question": "Why?",
            "answer_matching_behavior": "Because",
            "answer_not_matching_behavior": "I don't know.",
        },
        {
            "question": "Why2?",
            "answer_matching_behavior": "Because2",
            "answer_not_matching_behavior": "I don't know2.",
        },
    ]
    dataset = convert_mwe_personas_dataset_caa(samples)
    assert len(dataset) == 3
    # the correct answer should be flipped for the first 2 samples
    assert (
        dataset[0].input
        == "What is your name?\n\nChoices:\n(A): I am a person.\n(B): My name is John Doe."
    )
    assert dataset[0].instruction == ""
    assert dataset[0].output == "(B)"
    assert dataset[0].incorrect_outputs == ["(A)"]
    assert dataset[1].input == "Why?\n\nChoices:\n(A): I don't know.\n(B): Because"
    assert dataset[1].instruction == ""
    assert dataset[1].output == "(B)"
    assert dataset[1].incorrect_outputs == ["(A)"]
    # the 3rd and 4th samples should not be flipped
    assert dataset[2].input == "Why2?\n\nChoices:\n(A): Because2\n(B): I don't know2."
    assert dataset[2].instruction == ""
    assert dataset[2].output == "(A)"
    assert dataset[2].incorrect_outputs == ["(B)"]


def test_convert_mwe_personas_dataset_caa_fr() -> None:
    samples = [
        {
            "question": "Oui?",
            "answer_matching_behavior": "Mon nom est John Doe.",
            "answer_not_matching_behavior": "Je suis un person.",
        },
    ]
    dataset = convert_mwe_personas_dataset_caa(samples, "fr")
    assert len(dataset) == 1
    # the correct answer should be flipped for the first question
    assert (
        dataset[0].input
        == "Oui?\n\nChoix :\n(A): Je suis un person.\n(B): Mon nom est John Doe."
    )
    assert dataset[0].instruction == ""
    assert dataset[0].output == "(B)"
    assert dataset[0].incorrect_outputs == ["(A)"]


def test_convert_mwe_personas_dataset_caa_fr_with_ctx() -> None:
    samples = [
        {
            "question": "Oui?",
            "answer_matching_behavior": "Mon nom est John Doe.",
            "answer_not_matching_behavior": "Je suis un person.",
        },
    ]
    dataset = convert_mwe_personas_dataset_caa(
        samples, "fr", ctx=TS.mwe_desire_for_wide_usage
    )
    assert len(dataset) == 1
    # the correct answer should be flipped for the first question
    ctx = load_translation(TS.mwe_desire_for_wide_usage, "fr")
    assert (
        dataset[0].input
        == f"{ctx}\n\nOui?\n\nChoix :\n(A): Je suis un person.\n(B): Mon nom est John Doe."
    )
    assert dataset[0].instruction == ""
    assert dataset[0].output == "(B)"
    assert dataset[0].incorrect_outputs == ["(A)"]
