# brunello - seed 3

# syntax reg 0.001  seed 3
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --use_wandb --seed 3

# syntax reg 0.01  seed 3
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --use_wandb --seed 3

# syntax reg 0.001  seed 2
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --use_wandb --seed 2

# syntax reg 0.01  seed 2
python train.py --segmented_data data/cc100_1b_tokens_heads.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --use_wandb --seed 2

# normal  seed 3
python train.py --segmented_data data/cc100_1b_tokens.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0 --use_wandb --seed 3