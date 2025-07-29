#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# create pre-tokenized cc100 training data
python -m src.data.pretokenize -m gpt2 -d cc100 -t -v -ts 1_000_000_000 -vs 100_000 -to cc100_1b_tokens.jsol -vo cc100_100k_tokens.jsonl

# create dependency-parsed cc100 data
python -m src.data.make_dataset_with_head