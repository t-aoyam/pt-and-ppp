#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# no regularization  seed 3 
python -m src.training.train.py --segmented_data data/cc100_1b_tokens.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0 --use_wandb --seed 3

# syntax reg 0.001  seed 3

python -m src.training.train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --use_wandb --seed 3

# copy reg  0.001  seed 3
python -m src.training.train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --smooth --seed 3 --use_wandb