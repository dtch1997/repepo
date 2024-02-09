"""
Update this file to add more supported languages and styles.
"""

from typing import Literal


TranslatableLangCode = Literal["fr", "ja", "zh"]
LangCode = TranslatableLangCode | Literal["en"]
TranslatableStyleCode = Literal["pirate"]
StyleCode = TranslatableStyleCode | Literal["ctx"]
TranslatableLangOrStyleCode = TranslatableLangCode | TranslatableStyleCode
LangOrStyleCode = LangCode | StyleCode

TRANSLATABLE_LANG_MAPPING: dict[TranslatableLangCode, str] = {
    "fr": "French",
    "ja": "Japanese",
    "zh": "Simplified Chinese",
}
LANG_MAPPING: dict[LangCode, str] = {**TRANSLATABLE_LANG_MAPPING, "en": "English"}
TRANSLATABLE_STYLE_MAPPING: dict[TranslatableStyleCode, str] = {
    "pirate": "pirate",
}
STYLE_MAPPING: dict[StyleCode, str] = {
    **TRANSLATABLE_STYLE_MAPPING,
    "ctx": "contextual",
}
TRANSLATABLE_MAPPING: dict[TranslatableLangOrStyleCode, str] = {
    **TRANSLATABLE_LANG_MAPPING,
    **TRANSLATABLE_STYLE_MAPPING,
}
LANG_OR_STYLE_MAPPING: dict[LangOrStyleCode, str] = {
    **LANG_MAPPING,
    **STYLE_MAPPING,
}
