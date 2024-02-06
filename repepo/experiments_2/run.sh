#!/bin/bash
datasets_arg="${1:-dev}"
train_split_arg="${2:-train-dev}"
test_split_arg="${3:-val-dev}"

echo "Running experiments_2 with datasets: $datasets_arg, train_split: $train_split_arg, test_split: $test_split_arg"

python repepo/experiments_2/extract_activations.py \
    --datasets "$datasets_arg" \
    --train_split_name "$train_split_arg"

python repepo/experiments_2/extract_activations.py \
    --datasets "$datasets_arg" \
    --train_split_name "$train_split_arg"

python repepo/experiments_2/extract_concept_vectors.py \
    --datasets "$datasets_arg" \
    --train_split_name "$train_split_arg"

python repepo/experiments_2/evaluate_concept_vectors.py \
    --datasets "$datasets_arg" \
    --train_split_name "$train_split_arg" \
    --test_split_name "$test_split_arg"
