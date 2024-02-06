from repepo.data.multiple_choice.make_mwe_personas_caa import (
    convert_mwe_personas_dataset_caa,
)


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
    ]
    dataset = convert_mwe_personas_dataset_caa(samples)
    assert len(dataset) == 2
    # the correct answer should be flipped for the first question
    assert (
        dataset[0].input
        == "What is your name?\n\nChoices:\n(A): I am a person.\n(B): My name is John Doe."
    )
    assert dataset[0].instruction == ""
    assert dataset[0].output == "(B)"
    assert dataset[0].incorrect_outputs == ["(A)"]
    assert dataset[1].input == "Why?\n\nChoices:\n(A): Because\n(B): I don't know."
    assert dataset[1].instruction == ""
    assert dataset[1].output == "(A)"
    assert dataset[1].incorrect_outputs == ["(B)"]


def test_convert_mwe_personas_dataset_caa_fr() -> None:
    samples = [
        {
            "question": "Oui?",
            "answer_matching_behavior": "Mon nom est John Doe.",
            "answer_not_matching_behavior": "Je suis un person.",
        },
        {
            "question": "Pourquoi?",
            "answer_matching_behavior": "Parce-que",
            "answer_not_matching_behavior": "Je ne sais pas.",
        },
    ]
    dataset = convert_mwe_personas_dataset_caa(samples, "fr")
    assert len(dataset) == 2
    # the correct answer should be flipped for the first question
    assert (
        dataset[0].input
        == "Oui?\n\nChoix :\n(A): Je suis un person.\n(B): Mon nom est John Doe."
    )
    assert dataset[0].instruction == ""
    assert dataset[0].output == "(B)"
    assert dataset[0].incorrect_outputs == ["(A)"]
    assert (
        dataset[1].input == "Pourquoi?\n\nChoix :\n(A): Parce-que\n(B): Je ne sais pas."
    )
    assert dataset[1].instruction == ""
    assert dataset[1].output == "(A)"
    assert dataset[1].incorrect_outputs == ["(B)"]
