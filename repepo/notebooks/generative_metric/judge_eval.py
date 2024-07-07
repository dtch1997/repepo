# flake8: noqa
# %%
import os
import re
import dotenv

dotenv.load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None

import json
import pandas as pd
from openai import OpenAI

client = OpenAI()


def get_judge_rating(
    response: str,
) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": judge_prompts["corrigibility"]},
            {"role": "user", "content": response},
        ],
    )

    judge_rating = completion.choices[0].message.content
    assert judge_rating is not None
    return judge_rating


def parse_judge_rating(rating_str: str) -> int:
    """Return the first number in the string"""
    # Search the string for the first occurence of a number
    # And then return it
    result = re.search(r"\d+", rating_str)
    if result is None:
        return -1
    return int(result.group())


# %%
with open("/workspace/repepo/repepo/notebooks/judge_prompts.json", "r") as f:
    judge_prompts = json.load(f)

df = pd.read_csv("/workspace/repepo/repepo/notebooks/openness-steering.csv")
df["text"] = df["raw_prompt_gen"] + " " + df["response"]
df["judge_rating"] = df["text"].apply(get_judge_rating)

df.to_csv(
    "/workspace/repepo/repepo/notebooks/openness-steering-with-judge.csv", index=False
)

# %%
import pandas as pd
import seaborn as sns

sns.set_theme()
df = pd.read_csv("/workspace/repepo/repepo/notebooks/openness-steering-with-judge.csv")
df["judge_rating_int"] = df["judge_rating"].apply(parse_judge_rating)
# sns.scatterplot(data=df, x='judge_rating', y='logit_diff', hue='prompt_idx')
df.head()

# %%
sns.scatterplot(df, x="judge_rating_int", y="logit_diff")

# %%
