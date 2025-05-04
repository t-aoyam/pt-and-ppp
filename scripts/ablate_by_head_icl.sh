#!/bin/bash

pythias=("EleutherAI/pythia-410m" "EleutherAI/pythia-160m" "EleutherAI/pythia-70m")
#pythias=()
#gpt2s=("models/gpt2-mlp-2-layers/checkpoint-15625" "models/gpt2-mlp-1-layers/checkpoint-15625" "models/gpt2-mlp-2-layers/checkpoint-2441400" "models/gpt2-mlp-1-layers/checkpoint-2441406")
gpt2s=("models/gpt2-mlp-2-layers/checkpoint-31250" "models/gpt2-mlp-1-layers/checkpoint-31250" "models/gpt2-mlp-2-layers/checkpoint-62500" "models/gpt2-mlp-1-layers/checkpoint-62500")
revisions=(2000 3000 4000)

# Declare associative arrays
declare -A layer_map
declare -A head_map
declare -A data_map

# Fill the dictionaries
layer_map["EleutherAI/pythia-410m"]=24
layer_map["EleutherAI/pythia-160m"]=12
layer_map["EleutherAI/pythia-70m"]=6
layer_map["models/gpt2-mlp-2-layers/checkpoint-15625"]=2
layer_map["models/gpt2-mlp-1-layers/checkpoint-15625"]=1
layer_map["models/gpt2-mlp-2-layers/checkpoint-31250"]=2
layer_map["models/gpt2-mlp-1-layers/checkpoint-31250"]=1
layer_map["models/gpt2-mlp-2-layers/checkpoint-62500"]=2
layer_map["models/gpt2-mlp-1-layers/checkpoint-62500"]=1
layer_map["models/gpt2-mlp-2-layers/checkpoint-2441400"]=2
layer_map["models/gpt2-mlp-1-layers/checkpoint-2441406"]=1

head_map["EleutherAI/pythia-410m"]=16
head_map["EleutherAI/pythia-160m"]=12
head_map["EleutherAI/pythia-70m"]=8
head_map["models/gpt2-mlp-2-layers/checkpoint-15625"]=8
head_map["models/gpt2-mlp-1-layers/checkpoint-15625"]=8
head_map["models/gpt2-mlp-2-layers/checkpoint-31250"]=8
head_map["models/gpt2-mlp-1-layers/checkpoint-31250"]=8
head_map["models/gpt2-mlp-2-layers/checkpoint-62500"]=8
head_map["models/gpt2-mlp-1-layers/checkpoint-62500"]=8
head_map["models/gpt2-mlp-2-layers/checkpoint-2441400"]=8
head_map["models/gpt2-mlp-1-layers/checkpoint-2441406"]=8

data_map["EleutherAI/pythia-410m"]="data/pile_10m_tokens_pythia-01.pkl"
data_map["EleutherAI/pythia-160m"]="data/pile_10m_tokens_pythia-01.pkl"
data_map["EleutherAI/pythia-70m"]="data/pile_10m_tokens_pythia-01.pkl"
data_map["models/gpt2-mlp-2-layers/checkpoint-15625"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-1-layers/checkpoint-15625"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-2-layers/checkpoint-31250"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-1-layers/checkpoint-31250"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-2-layers/checkpoint-62500"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-1-layers/checkpoint-62500"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-2-layers/checkpoint-2441400"]="data/pile_10m_tokens-01.pkl"
data_map["models/gpt2-mlp-1-layers/checkpoint-2441406"]="data/pile_10m_tokens-01.pkl"


# Loop over number of layers (e.g., from 2 to 6)
for model in "${pythias[@]}"; do

	num_layers=$((${layer_map[$model]} - 1))
	num_heads=$((${head_map[$model]} - 1))
	data_fp=${data_map[$model]}
	for revision in "${revisions[@]}"; do
		for l in $(seq 0 $num_layers); do
			# Loop over number of heads (e.g., from 4 to 12 in steps of 4)
			for h in $(seq 0 $num_heads); do
				# Combine the numbers with a hyphen
				config="${l}-${h}"
				# Run the Python script with the combined argument
				python get_icl_scores.py -m "$model" -d "$data_fp" -r "$revision" -am 'pp' -ah "$config" -ds 100000 -p -e1 40 -e2 60 -l1 450 -l2 550
				# Optional: Add a separator between runs
				echo "Finished running model: $model"
				echo "config: $config"
				echo "data: $data_fp"
				echo "------------------------"
			done
		done
	done
done

# Loop over number of layers (e.g., from 2 to 6)
for model in "${gpt2s[@]}"; do

	num_layers=$((${layer_map[$model]} - 1))
	num_heads=$((${head_map[$model]} - 1))
	data_fp=${data_map[$model]}
	for l in $(seq 0 $num_layers); do
		# Loop over number of heads (e.g., from 4 to 12 in steps of 4)
		for h in $(seq 0 $num_heads); do
			# Combine the numbers with a hyphen
			config="${l}-${h}"
			# Run the Python script with the combined argument
			python get_icl_scores.py -m "$model" -d "$data_fp" -am 'pp' -ah "$config" -ds 100000 -p -e1 40 -e2 60 -l1 450 -l2 550
			# Optional: Add a separator between runs
			echo "Finished running model: $model"
			echo "config: $config"
			echo "data: $data_fp"
			echo "------------------------"
		done
	done
done