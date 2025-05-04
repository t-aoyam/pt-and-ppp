# chianti - seed 3
# copy reg  0.001
python train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --smooth --seed 3

# copy reg 0.01
python train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --smooth --seed 3

# syntax reg 0.001
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --use_wandb --seed 3

# syntax reg 0.01
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --use_wandb --seed 3

# normal
python train.py --segmented_data data/cc100_1b_tokens.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0 --use_wandb --seed 3