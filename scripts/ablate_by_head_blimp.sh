#!/bin/bash

pythias=("EleutherAI/pythia-70m" "EleutherAI/pythia-160m" "EleutherAI/pythia-410m")
gpt2s=("models/gpt2-mlp-2-layers/checkpoint-15625" "models/gpt2-mlp-1-layers/checkpoint-15625" "models/gpt2-mlp-2-layers/checkpoint-31250" "models/gpt2-mlp-1-layers/checkpoint-31250" "models/gpt2-mlp-2-layers/checkpoint-62500" "models/gpt2-mlp-1-layers/checkpoint-62500" "models/gpt2-mlp-2-layers/checkpoint-2441400" "models/gpt2-mlp-1-layers/checkpoint-2441406")
revisions=(256 512 1000 2000 3000)

# Declare associative arrays
declare -A layer_map
declare -A head_map

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

# Loop over number of layers (e.g., from 2 to 6)
for model in "${pythias[@]}"; do
	for revision in "${revisions[@]}"; do
		num_layers=$((${layer_map[$model]} - 1))
		num_heads=$((${head_map[$model]} - 1))
		for l in $(seq 0 $num_layers); do
			# Loop over number of heads (e.g., from 4 to 12 in steps of 4)
			for h in $(seq 0 $num_heads); do
				# Combine the numbers with a hyphen
				config="${l}-${h}"
				# Run the Python script with the combined argument
				python evaluate_on_blimp.py -m "$model" -r "$revision" -ah "$config" -am 'pp' -b 128
				# Optional: Add a separator between runs
				echo "Finished running model: $model"
				echo "config: $config"
				echo "------------------------"
			done
		done
	done
done

for model in "${gpt2s[@]}"; do
	ffload it to CPU temporarily
	num_layers=$((${layer_map[$model]} - 1))
	num_heads=$((${head_map[$model]} - 1))
	for l in $(seq 0 $num_layers); do
		# Loop over number of heads (e.g., from 4 to 12 in steps of 4)
		for h in $(seq 0 $num_heads); do
			# Combine the numbers with a hyphen
			config="${l}-${h}"
			# Run the Python script with the combined argument
			python evaluate_on_blimp.py -m "$model" -r "$revision" -ah "$config" -am 'pp'       
			# Optional: Add a separator between runs
			echo "Finished running model: $model"
			echo "config: $config"
			echo "------------------------"
		done
	done
done
