# copy - Brunello

# copy reg 0.01  seed 2 - 
python train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --smooth --seed 2 --use_wandb

# copy reg  0.001  seed 3
python train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.001 --smooth --seed 3 --use_wandb

# copy reg 0.01  seed 3
python train.py --tokenized_data data/cc100_1b_tokens.jsonl --regularization_data data/rep_seqs_1b.jsonl --config_fp data/configs/matching_steps.json --n_layer 2 --t_block mlp --reg_lambda 0.01 --smooth --seed 3 --use_wandb
