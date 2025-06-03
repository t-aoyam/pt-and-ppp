models = (
	"gpt2-mlp-l2-b4-n005-s1"
	"gpt2-mlp-l2-b4-n50-s1"
)

for model in "${models[@]}"; do
	model="models/$model"
	# Run the Python script with the combined argument
	python get_surprisal.py -m "$model"
	# Optional: Add a separator between runs
	echo "Finished running model: $model"
	echo "------------------------"
done
