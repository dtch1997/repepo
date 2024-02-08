from functools import partial
from .constants import LANG_MAPPING, STYLE_MAPPING, LangOrStyleCode
from .gpt4_translate import gpt4_language_translate, gpt4_style_translate
from .google_translate import google_translate_bulk
from .translate_strings_parallel import translate_strings_parallel


# TODO: have a more elegant way to handle different translation backends
def lang_or_style_translate(
    texts: set[str],
    lang_or_style: LangOrStyleCode,
    use_google_for_langs=True,
    gpt_model: str = "gpt-4-turbo-preview",
    gpt_max_tokens: int = 256,
    gpt_temperature: float = 0.7,
    gpt_stop=None,
    gpt_strip_outer_quotes: bool = True,
    gpt_worker_threads: int = 50,
    show_progress: bool = True,
) -> dict[str, str]:
    """
    Translate a string to a different style using GPT4 or Google.
    """
    translate_fn = None
    if lang_or_style in LANG_MAPPING:
        if use_google_for_langs:
            input_texts = list(texts)
            output_texts = google_translate_bulk(
                input_texts,
                target_language_code=lang_or_style,
                show_progress=show_progress,
            )
            return dict(zip(input_texts, output_texts))
        else:
            translate_fn = partial(
                gpt4_language_translate,
                lang_code=lang_or_style,
                model=gpt_model,
                max_tokens=gpt_max_tokens,
                temperature=gpt_temperature,
                stop=gpt_stop,
                strip_outer_quotes=gpt_strip_outer_quotes,
            )
    elif lang_or_style in STYLE_MAPPING:
        translate_fn = partial(
            gpt4_style_translate,
            style=lang_or_style,
            model=gpt_model,
            max_tokens=gpt_max_tokens,
            temperature=gpt_temperature,
            stop=gpt_stop,
            strip_outer_quotes=gpt_strip_outer_quotes,
        )
    else:
        raise ValueError(f"Unknown lang_or_style: {lang_or_style}")
    assert translate_fn is not None
    return translate_strings_parallel(
        texts, translate_fn, max_workers=gpt_worker_threads, show_progress=show_progress
    )
