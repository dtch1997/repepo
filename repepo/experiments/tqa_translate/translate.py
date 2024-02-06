""" Translate TQA and save as CSV """

import pandas as pd

from arrr import translate
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from repepo.data.make_dataset import make_dataset, DatasetSpec
from repepo.core.types import Example
from typing import Callable
from dataclasses import asdict

TranslationFn = Callable[[str], str]

vowels = ["a", "e", "i", "o", "u"]
punctuation = [".", ",", "!", "?", ";", ":"]


def translate_to_pig_latin(english: str) -> str:
    with MosesTokenizer("en") as tokenizer:
        english_words = tokenizer(english)

    latin_words = []

    for word in english_words:
        if len(word) == 0:
            continue

        # If the word is a punctuation, continue
        if word[0] in punctuation:
            latin_words.append(word)

        # If the word starts with a vowel then simply postfix "yay"
        elif word[0] in vowels:
            latin_words.append(word + "yay")

        # If the word starts with a consonant, then find the first vowel
        # and postfix all the consonants present before that vowel to the end
        # of the word and postfix "ay"
        else:
            has_vowel = False
            for i in range(len(word)):
                if word[i] in vowels:
                    latin_words.append(word[i:] + word[:i] + "ay")
                    has_vowel = True
                    break

            # If the word does not have a vowel, then postfix "ay"
            if not has_vowel:
                latin_words.append(word + "ay")

    with MosesDetokenizer("en") as detokenizer:
        return detokenizer(latin_words)


def translate_to_leetspeak(english: str) -> str:
    english_words = english.split(" ")
    leet_words = []

    for word in english_words:
        leet_word = ""
        for letter in word:
            if letter == "a":
                leet_word += "4"
            elif letter == "e":
                leet_word += "3"
            elif letter == "i":
                leet_word += "1"
            elif letter == "o":
                leet_word += "0"
            elif letter == "s":
                leet_word += "5"
            else:
                leet_word += letter
        leet_words.append(leet_word)

    return " ".join(leet_words)


def translate_to_pirate_speak(english: str) -> str:
    return translate(english)


def translate_example(example: Example, translation_func: TranslationFn) -> Example:
    if example.incorrect_outputs is None:
        return Example(
            instruction=translation_func(example.instruction),
            input=translation_func(example.input),
            output=translation_func(example.output),
            incorrect_outputs=None,
        )

    else:
        incorrect_outputs_trans = [
            translation_func(output) for output in example.incorrect_outputs
        ]
        return Example(
            instruction=translation_func(example.instruction),
            input=translation_func(example.input),
            output=translation_func(example.output),
            incorrect_outputs=incorrect_outputs_trans,
        )


if __name__ == "__main__":
    import pathlib

    current_dir = pathlib.Path(__file__).parent.absolute()

    list_data = make_dataset(
        DatasetSpec(
            name="truthfulqa",
        )
    )[:100]

    df_english = pd.json_normalize([asdict(example) for example in list_data])
    df_english.to_csv(current_dir / "truthfulqa_english.csv", index=False)

    # Translate to pig latin
    list_data_pig_latin = [
        translate_example(example, translate_to_pig_latin) for example in list_data
    ]
    df_pig_latin = pd.json_normalize(
        [asdict(example) for example in list_data_pig_latin]
    )
    df_pig_latin.to_csv(current_dir / "truthfulqa_pig_latin.csv", index=False)

    # Translate to leetspeak
    list_data_leetspeak = [
        translate_example(example, translate_to_leetspeak) for example in list_data
    ]
    df_leetspeak = pd.json_normalize(
        [asdict(example) for example in list_data_leetspeak]
    )
    df_leetspeak.to_csv(current_dir / "truthfulqa_leetspeak.csv", index=False)

    # Translate to pirate speak
    list_data_pirate_speak = [
        translate_example(example, translate_to_pirate_speak) for example in list_data
    ]
    df_pirate_speak = pd.json_normalize(
        [asdict(example) for example in list_data_pirate_speak]
    )
    df_pirate_speak.to_csv(current_dir / "truthfulqa_pirate_speak.csv", index=False)
