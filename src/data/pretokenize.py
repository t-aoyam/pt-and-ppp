from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset
import os, pathlib, json, tqdm
import numpy as np

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='EleutherAI/pythia-70m',
                        help='tokenizer, default=EleutherAI/pythia-70m')
    parser.add_argument('-d', '--data', default='EleutherAI/the_pile_deduplicated', # 'EleutherAI/the_pile'
                        help='data, default=EleutherAI/the_pile_deduplicated')
    parser.add_argument('-t', '--train', action='store_true',
                        help='whether or not to create train')
    parser.add_argument('-v', '--val', action='store_true',
                        help='whether or not to create val')
    parser.add_argument('-ts', '--train_size', type=int, default=1_000_000_000,
                        help='train size, default is 1B')
    parser.add_argument('-vs', '--val_size', type=int, default=100_000,
                        help='train size, default is 100K')
    parser.add_argument('-to', '--train_output_fn',
                        help='train output filename')
    parser.add_argument('-vo', '--val_output_fn',
                        help='val output filename')
    return parser.parse_args()

def tokenize(data, tokenizer, train_size, val_size, train_output_fn, val_output_fn):
    num_toks = 0
    train_toks = []
    train = data['train']
    with open(os.path.join(DATA_DIR, train_output_fn), 'w') as f:
        for sample in train:
            text = sample['text'].strip('\n').strip()
            if not text:
                train_toks.append(tokenizer.eos_token_id)
                continue

            outputs = tokenizer(
                text,
                truncation=False
            )
            train_toks.extend(outputs['input_ids'])
            print(f"\rTokenizing train portion (1B tokens)... {round(len(train_toks) / train_size, 5) * 100}%", end="")
            if len(train_toks) > 1024:
                f.write(
                    json.dumps(
                        {'input_ids': train_toks[:1024]}
                    ) + '\n'
                )
                train_toks = train_toks[1024:]
                num_toks += 1024
            if num_toks >= train_size:
                break
    train_toks.clear()

    val_toks = []  # discard training portion
    num_val_toks = 0
    if 'validation' in data:
        val = data['validation']
    else:
        val = train  # val set will be the train split, after 1B
    with open(os.path.join(DATA_DIR, val_output_fn), 'w') as f:
        for sample in val:
            text = sample['text'].strip('\n').strip()
            if not text:
                val_toks.append(tokenizer.eos_token_id)
                continue

            outputs = tokenizer(
                text,
                truncation=False
            )
            val_toks.extend(outputs['input_ids'])
            print(f"\rTokenizing validation portion (10M tokens)... {round(len(val_toks) / val_size, 5) * 100}%",
                  end="")
            if len(val_toks) >= 1024:
                f.write(
                    json.dumps(
                        {'input_ids': val_toks[:1024]}
                    ) + '\n'
                )
                val_toks = val_toks[1024:]
                num_val_toks += 1024
            if num_val_toks >= val_size:
                break
        val_toks.clear()
    return None

def main():
    args = parse_args()
    if 'cc100' in args.data:
        data = load_dataset(args.data, 'en', streaming=True)
    else:
        data = load_dataset(args.data, streaming=True)
    data = data['train']
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenize(data, tokenizer, args.train_size, args.val_size)

    return

if __name__ == "__main__":
    main()