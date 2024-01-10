import re
from collections import defaultdict
from typing import Callable, Iterable, Literal, Optional, Union

from torch import nn

from repepo.core.types import Model

LayerMatcher = Union[str, Callable[[nn.Module, int], str]]


def collect_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> list[str]:
    """
    Find all layers in the model that match the layer_matcher, in order by layer_num.
    layer_matcher can be a string formatted like "transformer.h.{num}.mlp" or a callable
    If layer_matcher is a callable, it should take in a model and layer_num and return
    a string representing the layer name corresponding to that layer number.
    If layer_matcher is a string, it's considered a template and MUST contain a "{num}" portion
    """
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    all_layer_names = dict(model.named_modules()).keys()
    matching_layers = []
    for layer_num, layer in enumerate(model.modules()):
        layer_name = matcher_callable(model, layer_num)
        if layer_name in all_layer_names:
            matching_layers.append(layer_name)
        else:
            break
    return matching_layers


def get_num_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> int:
    """Returns the number of layers in the model that match the layer_matcher"""
    return len(collect_matching_layers(model, layer_matcher))


def get_layer_name(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> str:
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    layer_num = fix_neg_layer_num(model, layer_matcher, layer_num)
    return matcher_callable(model, layer_num)


def fix_neg_layer_num(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> int:
    """Helper to handle negative layer nums. If layer_num is negative, return len(layers) + layer_num"""
    if layer_num >= 0:
        return layer_num
    matching_layers = collect_matching_layers(model, layer_matcher)
    return len(matching_layers) + layer_num


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    return dict(model.named_modules())[layer_name]


def _layer_matcher_to_callable(
    layer_matcher: LayerMatcher,
) -> Callable[[nn.Module, int], str]:
    if isinstance(layer_matcher, str):
        if "{num}" not in layer_matcher:
            raise ValueError(
                "layer_matcher must be a callable or a string containing {num}"
            )
        return lambda _model, layer_num: layer_matcher.format(num=layer_num)
    return layer_matcher


LAYER_GUESS_RE = r"^([^\d]+)\.([\d]+)(.*)$"


def guess_decoder_block_matcher(model: nn.Module) -> str | None:
    """
    Guess the hidden layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(dict(model.named_modules()).keys())


def guess_mlp_matcher(model: nn.Module) -> str | None:
    """
    Guess the mlp layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(), filter=lambda guess: "mlp" in guess
    )


def guess_self_attn_matcher(model: nn.Module) -> str | None:
    """
    Guess the self attention layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "attn" in guess or "attention" in guess,
    )


def guess_input_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the input layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_1" in guess or "input_layernorm" in guess,
    )


def guess_post_attention_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the post-attention layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_2" in guess or "post_attention_layernorm" in guess,
    )


# broken into a separate function for easier testing
def _guess_block_matcher_from_layers(
    layers: Iterable[str], filter: Optional[Callable[[str], bool]] = None
) -> str | None:
    counts_by_guess: dict[str, int] = defaultdict(int)

    for layer in layers:
        if re.match(LAYER_GUESS_RE, layer):
            guess = re.sub(LAYER_GUESS_RE, r"\1.{num}\3", layer)
            if filter is None or filter(guess):
                counts_by_guess[guess] += 1
    if len(counts_by_guess) == 0:
        return None

    # score is higher for guesses that match more often, are and shorter in length
    guess_scores = [
        (guess, count + 1 / len(guess)) for guess, count in counts_by_guess.items()
    ]
    return max(guess_scores, key=lambda x: x[1])[0]


LayerType = Literal[
    "decoder_block", "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"
]

ModelLayerConfig = dict[LayerType, LayerMatcher]

GptNeoxLayerConfig: ModelLayerConfig = {
    "decoder_block": "gpt_neox.layers.{num}",
    "self_attn": "gpt_neox.layers.{num}.attention",
    "mlp": "gpt_neox.layers.{num}.mlp",
    "input_layernorm": "gpt_neox.layers.{num}.input_layernorm",
    "post_attention_layernorm": "gpt_neox.layers.{num}.post_attention_layernorm",
}

LlamaLayerConfig: ModelLayerConfig = {
    "decoder_block": "model.layers.{num}",
    "self_attn": "model.layers.{num}.self_attn",
    "mlp": "model.layers.{num}.mlp",
    "input_layernorm": "model.layers.{num}.input_layernorm",
    "post_attention_layernorm": "model.layers.{num}.post_attention_layernorm",
}

Gpt2LayerConfig: ModelLayerConfig = {
    "decoder_block": "transformer.h.{num}",
    "self_attn": "transformer.h.{num}.attn",
    "mlp": "transformer.h.{num}.mlp",
    "input_layernorm": "transformer.h.{num}.ln_1",
    "post_attention_layernorm": "transformer.h.{num}.ln_2",
}


def check_predefined_layer_configs(model: Model) -> ModelLayerConfig | None:
    """Returns one of the above pre-defined layer configs if they match the model, else None"""
    for layer_config in [GptNeoxLayerConfig, LlamaLayerConfig, Gpt2LayerConfig]:
        everything_matches = True
        for layer_matcher in layer_config.values():
            if len(collect_matching_layers(model, layer_matcher)) == 0:
                everything_matches = False
                break
        if everything_matches:
            return layer_config
    return None


def enhance_model_config_matchers(
    model: Model, config: ModelLayerConfig
) -> ModelLayerConfig:
    """Returns a new layer config, attempting to fill-in missing layer matchers"""
    enhanced_config: ModelLayerConfig = {**config}
    if "decoder_block" not in config and (
        decoder_block_matcher := guess_decoder_block_matcher(model)
    ):
        enhanced_config["decoder_block"] = decoder_block_matcher
    if "mlp" not in config and (mlp_matcher := guess_mlp_matcher(model)):
        enhanced_config["mlp"] = mlp_matcher
    if "self_attn" not in config and (
        self_attn_matcher := guess_self_attn_matcher(model)
    ):
        enhanced_config["self_attn"] = self_attn_matcher
    if "input_layernorm" not in config and (
        input_layernorm_matcher := guess_input_layernorm_matcher(model)
    ):
        enhanced_config["input_layernorm"] = input_layernorm_matcher
    if "post_attention_layernorm" not in config and (
        post_attention_layernorm_matcher := guess_post_attention_layernorm_matcher(
            model
        )
    ):
        enhanced_config["post_attention_layernorm"] = post_attention_layernorm_matcher
    return enhanced_config


def guess_and_enhance_layer_config(
    model: Model, layer_config: Optional[ModelLayerConfig] = None
) -> ModelLayerConfig:
    """Try to guess any missing parts of the layer config, after checking against predefined configs"""
    if not layer_config:
        layer_config = check_predefined_layer_configs(model)
    layer_config = enhance_model_config_matchers(model, layer_config or {})
    return layer_config
