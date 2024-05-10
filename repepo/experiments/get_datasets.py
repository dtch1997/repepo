import seaborn as sns
import matplotlib.pyplot as plt
import random
from repepo.experiments.persona_prompts import (
    PERSONA_PROMPTS_BY_TOP_LEVEL_CATEGORIES,
    ADVANCED_AI_RISK,
    CAA_SYCOPHANCY_PROMPTS,
    CAA_TRUTHFULQA_PROMPTS,
)

random.seed(0)


# Sample 3 without replacement from each category
def get_sampled_persona_prompts(k: int = 3):
    sampled_personas = {}
    for name, category in PERSONA_PROMPTS_BY_TOP_LEVEL_CATEGORIES.items():
        keys = list(category.keys())
        sampled_keys = random.sample(keys, k)
        for key in sampled_keys:
            sampled_personas[key] = category[key]
    return sampled_personas


def get_all_prompts():
    sampled_persona_prompts = get_sampled_persona_prompts()
    return {
        **sampled_persona_prompts,
        **ADVANCED_AI_RISK,
        **CAA_SYCOPHANCY_PROMPTS,
        **CAA_TRUTHFULQA_PROMPTS,
    }


if __name__ == "__main__":
    sns.set_theme()
    sns.barplot(
        y=[name for name in PERSONA_PROMPTS_BY_TOP_LEVEL_CATEGORIES.keys()],
        x=[
            len(category)
            for category in PERSONA_PROMPTS_BY_TOP_LEVEL_CATEGORIES.values()
        ],
    )
    plt.title("Number of persona prompts by top-level category")
    plt.show()

    prompts = get_all_prompts()
    print(len(prompts))
    for key in prompts.keys():
        print(key)
