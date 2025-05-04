from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os, pathlib, json, pickle, tqdm
import numpy as np

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# data_size_for_lm = 10_000
# context_length = 1_024
# val_size = 1_000

data_size_for_lm = 1_000_000_000
context_length = 1_024
val_size = 10_000_000

# data = load_dataset('cc100', 'en', streaming=True)
data = load_dataset('EleutherAI/the_pile_deduplicated', streaming=True)
data = data['train']
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def tokenize_cc100(tokenizer):
    num_toks = 0
    # train_texts = []
    train_toks = []
    for sample in data:
        # train_texts.append(sample['text'])
        text = sample['text'].strip('\n').strip()
        if not text:
            train_toks.append(tokenizer.eos_token_id)
            continue

        outputs = tokenizer(
            text,
            truncation=False
        )
        train_toks.extend(outputs['input_ids'])
        print(f"\rTokenizing train portion (1B tokens)... {round(len(train_toks) / data_size_for_lm, 5) * 100}%", end="")
        if len(train_toks) >= data_size_for_lm:
            break

    # print(f"\nsaving training text...")
    # with open(os.path.join(DATA_DIR, 'cc100_1b_texts.json'), 'w') as f:
    #     json.dump(train_texts, f)
    # train_texts.clear()
    print(f"saving training token...\n")
    with open(os.path.join(DATA_DIR, 'cc100_1b_tokens.pkl'), 'wb') as f:
        pickle.dump(train_toks, f)
    # np.save(os.path.join(DATA_DIR, 'cc100_1b_tokens.npy'), np.array(train_toks))
    train_toks.clear()

    # val_texts = []
    val_toks = []  # discard training portion
    for sample in data:
        # val_texts.append(sample['text'])
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
        if len(val_toks) >= val_size:
            break
    # print(f"\nsaving val text...")
    # with open(os.path.join(DATA_DIR, 'cc100_10m_texts.json'), 'w') as f:
    #     json.dump(val_texts, f)

    # val_texts.clear()
    print(f"saving val token...")
    with open(os.path.join(DATA_DIR, 'cc100_10m_tokens.pkl'), 'wb') as f:
        pickle.dump(val_toks, f)
    val_toks.clear()
    return None

def tokenize_pile(tokenizer):
    pbar = tqdm.tqdm(total=data_size_for_lm, desc='Tokenizing train portion (1B tokens)...')
    with open(os.path.join(DATA_DIR, 'pile_1b_tokens_gpt.txt'), 'w') as f:
        num_toks = 0
        train_toks = []
        for sample in data:
            text = sample['text'].strip('\n').strip()
            if max([len(word) for word in text.split()]) > 150 or len(text) > 1_000_000:
                continue
            if not text:
                continue

            outputs = tokenizer(
                text,
                truncation=False
            )
            train_toks.extend(outputs['input_ids'])
            train_toks.append(tokenizer.eos_token_id)
            num_toks += len(train_toks)+1
            pbar.update(len(train_toks)+1)

            # print(f"\rTokenizing train portion (1B tokens)... {round(num_toks / data_size_for_lm, 5) * 100}%", end="")

            f.write('\n'.join(str(item) for item in train_toks))
            train_toks = ['']
            if num_toks >= data_size_for_lm:
                break
    # print(f"\nsaving training text...")
    # with open(os.path.join(DATA_DIR, 'cc100_1b_texts.json'), 'w') as f:
    #     json.dump(train_texts, f)
    # train_texts.clear()
    # print(f"saving training token...\n")
    # np.save(os.path.join(DATA_DIR, 'cc100_1b_tokens.npy'), np.array(train_toks))
    # train_toks.clear()
    #
    # # val_texts = []
    # val_toks = []  # discard training portion
    # for sample in data:
    #     # val_texts.append(sample['text'])
    #     text = sample['text'].strip('\n').strip()
    #     if not text:
    #         val_toks.append(tokenizer.eos_token_id)
    #         continue
    #
    #     outputs = tokenizer(
    #         text,
    #         truncation=False
    #     )
    #     val_toks.extend(outputs['input_ids'])
    #     print(f"\rTokenizing validation portion (10M tokens)... {round(len(val_toks) / val_size, 5) * 100}%",
    #           end="")
    #     if len(val_toks) >= val_size:
    #         break
    # # print(f"\nsaving val text...")
    # # with open(os.path.join(DATA_DIR, 'cc100_10m_texts.json'), 'w') as f:
    # #     json.dump(val_texts, f)
    #
    # # val_texts.clear()
    # print(f"saving val token...")
    # with open(os.path.join(DATA_DIR, 'cc100_10m_tokens.pkl'), 'wb') as f:
    #     pickle.dump(val_toks, f)
    # val_toks.clear()
    # return None

tokenize_pile(tokenizer)