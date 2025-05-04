#!/bin/bash

# pythias=("EleutherAI/pythia-410m" "EleutherAI/pythia-160m" "EleutherAI/pythia-70m")
# gpt2s=("models/gpt2-mlp-2-layers" "models/gpt2-mlp-1-layers" "models/gpt2-mlp-0-layers")
# gpt2s=("models/gpt2-mlp-l1-b4-r3" "models/gpt2-mlp-l2-b4-r3")
# gpt2s=("models/gpt2-mlp-l2-b16-r0" "models/gpt2-mlp-l2-b64-r0")
gpt2s=("models/gpt2-mlp-l2-b4-cir2-s1")

#for model in "${pythias[@]}"; do
#	# Run the Python script with the combined argument
#	python get_surprisal.py -m "$model"
#	# Optional: Add a separator between runs
#	echo "Finished running model: $model"
#	echo "------------------------"
#done

for model in "${gpt2s[@]}"; do
	# Run the Python script with the combined argument
	python get_surprisal.py -m "$model"
	# Optional: Add a separator between runs
	echo "Finished running model: $model"
	echo "------------------------"
done
