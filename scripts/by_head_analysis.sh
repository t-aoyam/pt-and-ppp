#!/bin/bash

#models=("EleutherAI-pythia-410m-step1000" "EleutherAI-pythia-160m-step1000" "EleutherAI-pythia-70m-step1000" "EleutherAI-pythia-410m-step2000" "EleutherAI-pythia-160m-step2000" "EleutherAI-pythia-70m-step2000" "EleutherAI-pythia-410m-step3000" "EleutherAI-pythia-160m-step3000" "EleutherAI-pythia-70m-step3000"  "EleutherAI-pythia-410m-step4000" "EleutherAI-pythia-160m-step4000" "EleutherAI-pythia-70m-step4000" "EleutherAI-pythia-410m-step5000" "EleutherAI-pythia-160m-step5000" "EleutherAI-pythia-70m-step5000" "gpt2-mlp-2-layers-checkpoint-15625" "gpt2-mlp-1-layers-checkpoint-15625" "gpt2-mlp-2-layers-checkpoint-31250" "gpt2-mlp-1-layers-checkpoint-31250" "gpt2-mlp-2-layers-checkpoint-62500" "gpt2-mlp-1-layers-checkpoint-62500" "gpt2-mlp-2-layers-checkpoint-2441400" "gpt2-mlp-1-layers-checkpoint-2441406")
models=("EleutherAI-pythia-160m-step256" "EleutherAI-pythia-160m-step512" "EleutherAI-pythia-410m-step256" "EleutherAI-pythia-410m-step512" "gpt2-mlp-2-layers-checkpoint-3906" "gpt2-mlp-1-layers-checkpoint-3906" "gpt2-mlp-2-layers-checkpoint-7812" "gpt2-mlp-1-layers-checkpoint-7812")
# Declare associative arrays
declare -A layer_map
declare -A head_map


# Fill the dictionaries
layer_map["EleutherAI-pythia-160m-step256"]=12
layer_map["EleutherAI-pythia-160m-step512"]=12
layer_map["EleutherAI-pythia-410m-step256"]=24
layer_map["EleutherAI-pythia-410m-step512"]=24
layer_map["EleutherAI-pythia-410m-step1000"]=24
layer_map["EleutherAI-pythia-160m-step1000"]=12
layer_map["EleutherAI-pythia-70m-step1000"]=6
layer_map["EleutherAI-pythia-410m-step2000"]=24
layer_map["EleutherAI-pythia-160m-step2000"]=12
layer_map["EleutherAI-pythia-70m-step2000"]=6
layer_map["EleutherAI-pythia-410m-step3000"]=24
layer_map["EleutherAI-pythia-160m-step3000"]=12
layer_map["EleutherAI-pythia-70m-step3000"]=6
layer_map["EleutherAI-pythia-410m-step4000"]=24
layer_map["EleutherAI-pythia-160m-step4000"]=12
layer_map["EleutherAI-pythia-70m-step4000"]=6
layer_map["EleutherAI-pythia-410m-step5000"]=24
layer_map["EleutherAI-pythia-160m-step5000"]=12
layer_map["EleutherAI-pythia-70m-step5000"]=6
layer_map["gpt2-mlp-2-layers-checkpoint-3906"]=2
layer_map["gpt2-mlp-1-layers-checkpoint-3906"]=1
layer_map["gpt2-mlp-2-layers-checkpoint-7812"]=2
layer_map["gpt2-mlp-1-layers-checkpoint-7812"]=1
layer_map["gpt2-mlp-2-layers-checkpoint-62500"]=2
layer_map["gpt2-mlp-1-layers-checkpoint-62500"]=1
layer_map["gpt2-mlp-2-layers-checkpoint-2441400"]=2
layer_map["gpt2-mlp-1-layers-checkpoint-2441406"]=1

head_map["EleutherAI-pythia-160m-step256"]=12
head_map["EleutherAI-pythia-160m-step512"]=12
head_map["EleutherAI-pythia-410m-step256"]=16
head_map["EleutherAI-pythia-410m-step512"]=16
head_map["EleutherAI-pythia-70m-step256"]=8
head_map["EleutherAI-pythia-70m-step512"]=8
head_map["EleutherAI-pythia-410m-step1000"]=16
head_map["EleutherAI-pythia-160m-step1000"]=12
head_map["EleutherAI-pythia-70m-step1000"]=8
head_map["EleutherAI-pythia-410m-step2000"]=16
head_map["EleutherAI-pythia-160m-step2000"]=12
head_map["EleutherAI-pythia-70m-step2000"]=8
head_map["EleutherAI-pythia-410m-step3000"]=16
head_map["EleutherAI-pythia-160m-step3000"]=12
head_map["EleutherAI-pythia-70m-step3000"]=8
head_map["EleutherAI-pythia-410m-step4000"]=16
head_map["EleutherAI-pythia-160m-step4000"]=12
head_map["EleutherAI-pythia-70m-step4000"]=8
head_map["EleutherAI-pythia-410m-step5000"]=16
head_map["EleutherAI-pythia-160m-step5000"]=12
head_map["EleutherAI-pythia-70m-step5000"]=8
head_map["gpt2-mlp-2-layers-checkpoint-3906"]=8
head_map["gpt2-mlp-1-layers-checkpoint-3906"]=8
head_map["gpt2-mlp-2-layers-checkpoint-7812"]=8
head_map["gpt2-mlp-1-layers-checkpoint-7812"]=8
head_map["gpt2-mlp-2-layers-checkpoint-62500"]=8
head_map["gpt2-mlp-1-layers-checkpoint-62500"]=8
head_map["gpt2-mlp-2-layers-checkpoint-2441400"]=8
head_map["gpt2-mlp-1-layers-checkpoint-2441406"]=8


# Loop over number of layers (e.g., from 2 to 6)
for model in "${models[@]}"; do

	num_layers=$((${layer_map[$model]} - 1))
	num_heads=$((${head_map[$model]} - 1))

	for l in $(seq 0 $num_layers); do
		# Loop over number of heads (e.g., from 4 to 12 in steps of 4)
		for h in $(seq 0 $num_heads); do
			# Combine the numbers with a hyphen
			config="${l}-${h}"
			# Run the Python script with the combined argument
			Rscript by_head_analysis.R "$model" "pp" "$config"       
			# Optional: Add a separator between runs
			echo "Finished computing the PPP of model: $model"
			echo "config: $config"
			echo "------------------------"
		done
	done
done