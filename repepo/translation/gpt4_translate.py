from functools import cache
import openai
from .constants import STYLE_MAPPING, LANG_MAPPING, LangCode, StyleCode


LANGUAGE_TRANSLATE_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that translates English text into {language}. You preserve the meaning of the text, as well as it's approximate length. You rewrite the text such that if you translated it back to English, it would produce the original text."""
STYLE_TRANSLATE_SYSTEM_PROMPT = """You are a helpful assistant that rewrites text in target styles. You preserve the meaning of the text, as well as it's approximate length. You rewrite the text such that if you translated it back to the original style, it would produce the original text."""


# putting this into a function so it doesn't error on import if not API key is set
@cache
def get_openai_client():
    """
    Expects OPENAI_API_KEY to be set in the environment.
    """
    return openai.OpenAI()


def _prompt_openai(
    system_prompt: str,
    prompt: str,
    model="gpt-4-turbo-preview",
    max_tokens=100,
    temperature=0.7,
    stop=None,
) -> str:
    """
    Generate a response from OpenAI's Chat Completion API.
    Based on the translators.ipynb notebook in examples
    """
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=stop,
    )
    if response.choices[0].message.content is None:
        raise ValueError(f"Prompting OpenAI failed for prompt: {prompt}")
    return response.choices[0].message.content.strip()


def _strip_outer_quotes(text: str) -> str:
    """
    GPT4 likes to add quotes around the text sometimes, so we strip them off.
    """
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def gpt4_style_translate(
    text: str,
    style: StyleCode,
    model="gpt-4-turbo-preview",
    max_tokens=100,
    temperature=0.7,
    stop=None,
    strip_outer_quotes=True,
) -> str:
    """
    Translate a string to a different style using GPT4.
    Based on the translators.ipynb notebook in examples
    """

    # Define the prompt for the translation task
    style_name = STYLE_MAPPING[style]
    prompt = f'Translate the following text into {style_name} style: "{text}"'
    res = _prompt_openai(
        system_prompt=STYLE_TRANSLATE_SYSTEM_PROMPT,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    if strip_outer_quotes:
        res = _strip_outer_quotes(res)
    return res


def gpt4_language_translate(
    text: str,
    lang_code: LangCode,
    model="gpt-4-turbo-preview",
    max_tokens=100,
    temperature=0.7,
    stop=None,
    strip_outer_quotes=True,
) -> str:
    """
    Translate a string to a different style using GPT4.
    Based on the translators.ipynb notebook in examples
    """
    lang_name = LANG_MAPPING[lang_code]
    system_prompt = LANGUAGE_TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language=lang_name)
    res = _prompt_openai(
        system_prompt=system_prompt,
        prompt=text,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    if strip_outer_quotes:
        res = _strip_outer_quotes(res)
    return res
