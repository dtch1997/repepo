import torch
from repepo.core.hook import SteeringHook

from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Dataset, Example, Tokenizer, Completion
from repepo.core.pipeline import Pipeline, PipelineContext

from repepo.steering.utils.helpers import EmptyTorchCUDACache
from repepo.steering.run_experiment import (
    build_steering_vector_training_data,
    extract_activations,
    get_aggregator,
    SteeringVector,
)

from steering_vectors import guess_and_enhance_layer_config
from steering_vectors.train_steering_vector import aggregate_activations

from transformers import LlamaForCausalLM
from tests._original_caa.llama_wrapper import LlamaWrapper


def test_get_steering_vector_matches_caa(
    empty_llama_model: LlamaForCausalLM, llama_chat_tokenizer: Tokenizer
):
    model = empty_llama_model
    tokenizer = llama_chat_tokenizer
    pipeline = Pipeline(
        model,
        tokenizer,
        formatter=LlamaChatFormatter(),
    )

    ####
    # First, calculate our SV
    dataset: Dataset = [
        Example(
            positive=Completion("Paris is in", "France"),
            negative=Completion("Paris is in", "Germany"),
            steering_token_index=-2,
        ),
    ]
    steering_vector_training_data = build_steering_vector_training_data(
        pipeline, dataset
    )

    layers = [0, 1, 2]

    # Extract activations
    with EmptyTorchCUDACache():
        pos_acts, neg_acts = extract_activations(
            pipeline.model,
            pipeline.tokenizer,
            steering_vector_training_data,
            layers=layers,
            show_progress=True,
            move_to_cpu=True,
        )

    # Aggregate activations
    aggregator = get_aggregator("mean")
    with EmptyTorchCUDACache():
        agg_acts = aggregate_activations(
            pos_acts,
            neg_acts,
            aggregator,
        )
        steering_vector = SteeringVector(
            layer_activations=agg_acts,
            layer_type="decoder_block",
        )

    # hackily translated from generate_vectors.py script
    tokenized_data = [
        (tokenizer.encode(svtd.positive_str), tokenizer.encode(svtd.negative_str))
        for svtd in steering_vector_training_data
    ]
    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])
    wrapped_model = LlamaWrapper(model, tokenizer)

    for p_tokens, n_tokens in tokenized_data:
        p_tokens = torch.tensor(p_tokens).unsqueeze(0).to(model.device)
        n_tokens = torch.tensor(n_tokens).unsqueeze(0).to(model.device)
        wrapped_model.reset_all()
        wrapped_model.get_logits(p_tokens)
        for layer in layers:
            p_activations = wrapped_model.get_last_activations(layer)
            p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        wrapped_model.reset_all()
        wrapped_model.get_logits(n_tokens)
        for layer in layers:
            n_activations = wrapped_model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    caa_vecs_by_layer = {}
    for layer in layers:
        all_pos_layer = torch.stack(pos_activations[layer])
        all_neg_layer = torch.stack(neg_activations[layer])
        caa_vecs_by_layer[layer] = (all_pos_layer - all_neg_layer).mean(dim=0)

    for layer in layers:
        assert torch.allclose(
            steering_vector.layer_activations[layer],
            caa_vecs_by_layer[layer],
            atol=1e-5,
        ), f"Non-matching activations at layer {layer}"


def test_evaluate_steering_vector_matches_caa_llama_wrapper(
    empty_llama_model: LlamaForCausalLM, llama_chat_tokenizer: Tokenizer
) -> None:
    model = empty_llama_model
    tokenizer = llama_chat_tokenizer
    pipeline = Pipeline(
        model,
        tokenizer,
        formatter=LlamaChatFormatter(),
    )
    test_example = Example(
        positive=Completion("Paris is in", "France"),
        negative=Completion("Paris is in", "Germany"),
        steering_token_index=-2,
    )

    layers = [0, 1, 2]
    multiplier = 7

    # Create a dummy SV
    steering_vector = SteeringVector(
        layer_activations={
            layer: torch.randn(1024, device=model.device) for layer in layers
        }
    )
    hook = SteeringHook(
        steering_vector=steering_vector,
        direction_multiplier=multiplier,
        patch_generation_tokens_only=True,
        skip_first_n_generation_tokens=1,
        layer_config=guess_and_enhance_layer_config(pipeline.model),
    )

    # hackily recreating what the pipeline does during logprobs
    base_prompt = pipeline.build_generation_prompt(test_example.positive)
    full_prompt = pipeline.build_full_prompt(test_example.positive)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    ctx = PipelineContext(
        method="logprobs",
        base_prompt=base_prompt,
        full_prompt=full_prompt,
        inputs=inputs,
        pipeline=pipeline,
    )
    orig_logits = model(**inputs).logits
    with hook(ctx):
        our_logits = model(**inputs).logits

    assert isinstance(hook, SteeringHook)  # keep pyright happy
    wrapped_model = LlamaWrapper(model, tokenizer, add_only_after_end_str=True)
    wrapped_model.reset_all()
    for layer in layers:
        wrapped_model.set_add_activations(
            layer, multiplier * hook.steering_vector.layer_activations[layer]
        )

    caa_logits = wrapped_model.get_logits(inputs["input_ids"].to(model.device))
    # only the final answer tokens should be different
    assert torch.allclose(our_logits[0, :-2], orig_logits[0, :-2])
    assert torch.allclose(our_logits, caa_logits)
