from .translation_strings import TranslationString, load_all_translations
from .constants import LangOrStyleCode

# convenience type for the translation string
TS = TranslationString


def load_translation(string: TS, lang_or_style: LangOrStyleCode | None = None) -> str:
    """Load a translation string from the system"""
    if lang_or_style is None:
        return string.value
    translations = load_all_translations()
    if (
        lang_or_style not in translations
        or string.name not in translations[lang_or_style]
    ):
        raise ValueError(
            f"Translation not found for {string.name} in {lang_or_style}. Please run `update_translations`."
        )
    return translations[lang_or_style][string.name]
