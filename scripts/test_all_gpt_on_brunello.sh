
models=(
#	"gpt2-mlp-l2-b4-n50-s1"
#	"gpt2-mlp-l2-b4-cir2-s2"
#	"gpt2-mlp-l2-b4-cir2-s3"
	"gpt2-mlp-l2-b4-cir3-s3"
#	"gpt2-mlp-l2-b4-r2-s3"
#	"gpt2-mlp-l2-b4-r3-s3"
	)

for model in "${models[@]}"; do
	model="models/$model"
	# Run the Python script with the combined argument
	python sas_probe.py -m "$model" -b 2 -u
	python sas_head.py -m "$model"

	# BLiMP
	python evaluate_on_blimp.py -m "$model"

	# PS
	python head_detector.py -m "$model"

	# ICL and loss
	python get_icl_scores.py -m "$model" -d "data/pile_10m_tokens-01.pkl" -ds 100000 -p -e1 40 -e2 60 -l1 450 -l2 550

	# surprisal
	python get_surprisal.py -m "$model"

	echo "Finished running model: $model"
	echo "------------------------"
done
