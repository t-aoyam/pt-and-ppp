#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# induction head
python -m evaluation.head_detector -m models/gpt2-mlp-l2-b4-cir2-s1/

# SAS
python sas_probe -m models/gpt2-mlp-l2-b4-cir2-s1/ -b 2

# surprisal
python -m evaluation.get_surprisal -m "$model"

# 