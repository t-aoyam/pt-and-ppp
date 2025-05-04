#!/bin/bash

#pythias=("EleutherAI/pythia-410m" "EleutherAI/pythia-160m" "EleutherAI/pythia-70m")
#gpt2s=("models/gpt2-mlp-2-layers" "models/gpt2-mlp-1-layers" "models/gpt2-mlp-0-layers")
# gpt2s=("models/gpt2-mlp-l2-b16-r0")
# gpt2s=("models/gpt2-mlp-l2-b64-r0")
gpt2s=("models/gpt2-mlp-l2-b4-cir2-s1")

# Declare associative arrays
declare -A data_map

# Fill the dictionaries
#data_map["EleutherAI/pythia-410m"]="data/pile_10m_tokens_pythia-01.pkl"
#data_map["EleutherAI/pythia-160m"]="data/pile_10m_tokens_pythia-01.pkl"
#data_map["EleutherAI/pythia-70m"]="data/pile_10m_tokens_pythia-01.pkl"
#data_map["models/gpt2-mlp-2-layers"]="data/pile_10m_tokens-01.pkl"
#data_map["models/gpt2-mlp-1-layers"]="data/pile_10m_tokens-01.pkl"
#data_map["models/gpt2-mlp-0-layers"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-l2-b4-cir2"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-l2-b64-r0"]="data/pile_10m_tokens-01.pkl"

# Loop over number of layers (e.g., from 2 to 6)
#for model in "${pythias[@]}"; do
#	data_fp=${data_map[$model]}
#	python get_icl_scores.py -m "$model" -d "$data_fp" -ds 100000 -p -e1 40 -e2 60 -l1 450 -l2 550
#	echo "Finished running model: $model"
#	echo "data: $data_fp"
#	echo "------------------------"
#done

# Loop over number of layers (e.g., from 2 to 6)
for model in "${gpt2s[@]}"; do
	data_fp=${data_map[$model]}
	# Run the Python script with the combined argument
	python get_icl_scores.py -m "$model" -d "$data_fp" -ds 100000 -p -e1 40 -e2 60 -l1 450 -l2 550
	# Optional: Add a separator between runs
	echo "Finished running model: $model"
	echo "data: $data_fp"
	echo "------------------------"
done
