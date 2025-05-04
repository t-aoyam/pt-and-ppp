#!/bin/bash

models=(
#	"EleutherAI-pythia-410m"
#	"EleutherAI-pythia-160m"
#	"EleutherAI-pythia-70m"
#	"gpt2-mlp-2-layers"
#	"gpt2-mlp-1-layers"
#	"gpt2-mlp-0-layers"
#	"gpt2-mlp-l2-b16-r0"
#	"gpt2-mlp-l2-b64-r0"
#	"gpt2-mlp-l1-b4-r3"
#	"gpt2-mlp-l2-b4-r3"
#	"gpt2-mlp-l2-b4-r2"
#	"gpt2-mlp-l2-b4-cir3"
#	"gpt2-mlp-l2-b4-cir2"
	"gpt2-mlp-l2-b4-cir2-s1"
	"gpt2-mlp-l2-b4-r2-s1"
	)

for model in "${models[@]}"; do
	# Run the Python script with the combined argument
	Rscript by_head_analysis.R "$model" NULL NULL
	# Optional: Add a separator between runs
	echo "Finished computing the PPP of model: $model"
	echo "------------------------"
done