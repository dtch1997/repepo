from typing import Sequence
from google.cloud import translate
from tqdm import tqdm

from repepo.variables import Environ


def google_translate_text(
    text: str, target_language_code: str, source_language_code: str = "en-US"
) -> str:
    return google_translate_bulk([text], target_language_code, source_language_code)[0]


def google_translate_bulk(
    texts: Sequence[str],
    target_language_code: str,
    source_language_code: str = "en-US",
    batch_size: int = 20,
    show_progress: bool = True,
) -> list[str]:
    client = translate.TranslationServiceClient()
    parent = f"projects/{Environ.GcpProjectId}/locations/global"
    results = []
    for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
        batch = texts[i : i + batch_size]
        batch_res = client.translate_text(
            request={
                "parent": parent,
                "contents": batch,
                "mime_type": "text/plain",
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        results.extend([res.translated_text for res in batch_res.translations])

    return results
