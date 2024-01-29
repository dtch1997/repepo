from arrr import translate

vowels = ["a", "e", "i", "o", "u"]

# TODO: Use sentencepiece for tokenization


def translate_to_pig_latin(english_words: list[str]) -> list[str]:
    latin_words = []

    for word in english_words:
        has_vowel = False

        for i in range(len(word)):
            """
            if first letter is a vowel
            """
            if word[0] in vowels:
                latin_words.append(word + "-yay")
                break

            else:
                """
                else get vowel position and postfix all the consonants
                present before that vowel to the end of the word along with "ay"
                """

                if word[i] in vowels:
                    latin_words.append(word[i:] + word[:i] + "ay")
                    has_vowel = True
                    break

                # if the word doesn't have any vowel then simply postfix "ay"
                if (not has_vowel) and i == len(word) - 1:
                    latin_words.append(word + "ay")
                    break

        raise RuntimeError("Should never reach here")

    return latin_words


def translate_to_leetspeak(english_words: list[str]) -> list[str]:
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

    return leet_words


def translate_to_pirate_speak(english_words: list[str]) -> list[str]:
    english = " ".join(english_words)
    pirate = translate(english)
    return pirate.split(" ")
