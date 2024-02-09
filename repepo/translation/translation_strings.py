"""
Update this file with new strings to be translated into all supported languages and styles.
"""

import argparse
from enum import Enum
from functools import cache
import json
from pathlib import Path
from .constants import TRANSLATABLE_MAPPING
from .lang_or_style_translate import lang_or_style_translate


TRANSLATED_STRINGS_FILE = Path(__file__).parent / "translated_strings.json"


# Add new strings here to be translated and used through the system
class TranslationString(Enum):
    llama2_chat_caa_system_message = "You are a helpful, honest and concise assistant."
    caa_choices = "Choices:"
    caa_eval_prefix = "My answer is"


@cache
def load_all_translations() -> dict[str, dict[str, str]]:
    """Get the current translations"""
    if TRANSLATED_STRINGS_FILE.exists():
        with open(TRANSLATED_STRINGS_FILE, "r") as jsonfile:
            return json.load(jsonfile)
    return {}


def update_translations(force: bool = False) -> None:
    """Update translations using gpt4"""
    current_translations = load_all_translations()
    for string in TranslationString:
        for lang_or_style in TRANSLATABLE_MAPPING:
            if lang_or_style not in current_translations:
                current_translations[lang_or_style] = {}
            if current_translations[lang_or_style].get(string.name) and not force:
                continue
            translation = list(
                lang_or_style_translate({string.value}, lang_or_style).values()
            )[0]
            current_translations[lang_or_style][string.name] = translation
    with open(TRANSLATED_STRINGS_FILE, "w") as jsonfile:
        json.dump(current_translations, jsonfile, indent=2, ensure_ascii=False)
    load_all_translations.cache_clear()


# TODO: we should have an explicit "scripts" folder for actually running scripts
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Force update translations"
    )
    args = parser.parse_args()

    update_translations(force=args.force)
    print("Updated translations")
