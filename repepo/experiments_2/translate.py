import openai

from dataclasses import dataclass
from repepo.core.types import Example, Dataset
from repepo.data.make_dataset import make_dataset, DatasetSpec
from concurrent.futures import ThreadPoolExecutor

SYSTEM_PROMPT = "You are a helpful assistant."
BETTER_SYSTEM_PROMPT = """You are a helpful assistant that rewrites text in target styles. You preserve the meaning of the text, as well as it's approximate length. You rewrite the text such that if you translated it back to the original style, it would produce the original text."""


def get_client():
    return openai.OpenAI()


@dataclass
class TranslationSettings:
    model: str = "gpt-4-turbo-preview"
    system_prompt: str = SYSTEM_PROMPT
    max_tokens: int = 100
    temperature: float = 0.7
    n: int = 1
    stop: str = None


def translate_to_style(
    client: openai.OpenAI, text: str, style: str, settings: TranslationSettings
) -> list[str]:
    """Translate text to given style.

    A wrapper around the OpenAI Chat Completions API"""
    # Define the prompt for the translation task
    prompt = f'Translate the following text into {style} style: "{text}"'

    # Generate the translation using Chat Completion API
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        n=settings.n,
        stop=settings.stop,
    )

    # Extract the translated text from the API response
    translation = [choice.message.content.strip() for choice in response.choices]
    return translation


def translate_example(
    client: openai.OpenAI, example: Example, style: str, settings
) -> Example:
    """Translate the input and output of an example to a different style"""
    # For now: skip the instruction as it's usually empty
    # For now: translate only the first incorrect output to save API calls
    with ThreadPoolExecutor() as executor:
        input_future = executor.submit(
            translate_to_style, client, example.input, style, settings
        )
        output_future = executor.submit(
            translate_to_style, client, example.output, style, settings
        )
        incorrect_future = executor.submit(
            translate_to_style, client, example.incorrect_outputs[0], style, settings
        )

        input_translation = input_future.result()[0]
        output_translation = output_future.result()[0]
        incorrect_translation = incorrect_future.result()[0]

    return Example(
        instruction="",
        input=input_translation,
        output=output_translation,
        incorrect_outputs=[incorrect_translation],
    )


def translate_dataset(
    client: openai.OpenAI, dataset: Dataset, style: str, settings: TranslationSettings
) -> Dataset:
    # Define a helper function to translate a single example
    def _translate_example_helper(example):
        return translate_example(client, example, style, settings)

    with ThreadPoolExecutor() as executor:
        translated_examples = executor.map(_translate_example_helper, dataset)

    return list(translated_examples)


if __name__ == "__main__":
    # Define the settings for the translation
    dataset_name = "subscribes-to-virtue-ethics"
    style = "pirate"
    settings = TranslationSettings(
        model="gpt-4-turbo-preview",
        system_prompt=BETTER_SYSTEM_PROMPT,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
    )

    # Create a client
    client = get_client()

    # Load the dataset
    dataset = make_dataset(DatasetSpec(name=dataset_name, split=":1%"))

    # Translate the dataset
    translated_dataset = translate_dataset(client, dataset, style, settings)

    print(translated_dataset[0])
