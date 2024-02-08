from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from tqdm import tqdm


def translate_strings_parallel(
    strings: set[str],
    translate_fn: Callable[[str], str],
    max_workers: int = 10,
    show_progress: bool = True,
    tqdm_desc: str = "Translating",
) -> dict[str, str]:
    """
    Translate a set of strings by applying a translate_fn in a threadpool.
    """
    translated_strings: dict[str, str] = {}
    original_strs = list(strings)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        translate_futures_to_orig = {
            executor.submit(translate_fn, original): original
            for original in original_strs
        }
        for future in tqdm(
            as_completed(translate_futures_to_orig),
            disable=not show_progress,
            desc=tqdm_desc,
            total=len(original_strs),
        ):
            original_str = translate_futures_to_orig[future]
            translated_strings[original_str] = future.result()
    return translated_strings
