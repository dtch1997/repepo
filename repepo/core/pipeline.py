from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import GenerationConfig

from .types import Example, Model, Tokenizer
from .prompt import Prompter, IdentityPrompter
from .format import Formatter, InputOutputFormatter


@dataclass
class Pipeline:
    """Generation pipeline"""

    model: Model
    tokenizer: Tokenizer
    prompter: Prompter = field(default_factory=IdentityPrompter)
    formatter: Formatter = field(default_factory=InputOutputFormatter)

    def build_generation_prompt(self, example: Example) -> str:
        """Build a prompt for generation"""
        completion = self.formatter.apply(example)
        completion = self.prompter.apply(completion)
        return completion.prompt

    def generate(
        self,
        example: Example,
        generation_config: Optional[GenerationConfig] = None,
        remove_base_prompt: bool = True,
    ) -> str:
        """Generate a completion for a given example"""
        base_prompt = self.build_generation_prompt(example)
        inputs: Any = self.tokenizer(base_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
        )[0]
        outputs_str = self.tokenizer.decode(outputs, skip_special_tokens=True)
        if remove_base_prompt:
            return outputs_str.replace(base_prompt, "")
        return outputs_str
