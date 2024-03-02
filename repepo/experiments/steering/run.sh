#!/bin/bash
configs_arg="${1:-dev}"
train_split_arg="${2:-train-dev}"
test_split_arg="${3:-val-dev}"

echo "Running steering experiments with configs: $configs_arg, train_split: $train_split_arg, test_split: $test_split_arg"

python repepo/experiments/steering/extract_activations.py \
    --configs "$configs_arg" \
    --train_split_name "$train_split_arg" \
    --test_split_name "$test_split_arg"    

python repepo/experiments/steering/compute_concept_metrics.py \
    --configs "$configs_arg" \
    --train_split_name "$train_split_arg" \
    --test_split_name "$test_split_arg"

python repepo/experiments/steering/aggregate_activations.py \
    --configs "$configs_arg" \
    --train_split_name "$train_split_arg" \
    --test_split_name "$test_split_arg"

python repepo/experiments/steering/apply_concept_vectors.py \
    --configs "$configs_arg" \
    --train_split_name "$train_split_arg" \
    --test_split_name "$test_split_arg"
