import os, tqdm
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from datasets import load_dataset
from transformers import AutoTokenizer

# data_fp = os.path.join('data', 'pile_1b_tokens_heads.jsonl')
# data_fp = os.path.join('data', 'cc100_1b_tokens_heads.jsonl')
data_fp = os.path.join('data', 'cc100_1b_tokens_heads.jsonl')
data = load_dataset('json', data_files=data_fp, streaming=True, split='train')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

pbar = tqdm.tqdm(total=int(800_000_000/1024))
doc = 0
for d in data:
    assert len(d['input_ids']) == len(d['tok2word']) == len(d['heads']) == 1024
    # assert d['tok2word'][-1] == d['heads'].index(-1)-1

    prev_word_i = 0
    words = []
    word = []
    for i, idx in enumerate(d['input_ids']):
        curr_word_i = d['tok2word'][i]
        if curr_word_i == prev_word_i:
            word.append(idx)
        else:
            prev_word_i = curr_word_i
            words.append(tokenizer.decode(word))
            word = [idx]
    words.append(tokenizer.decode(word))

    # print(tokenizer.decode(d['input_ids']))
    for i, h in enumerate(d['heads']):
        try:
            # span = sorted([i, int(h)-1])
            span = sorted([i, h])
            print(words[span[0]:span[1]+1], f"head: {words[h]} ||| child: {words[i]}")
        except:
            pass
    doc += 1
    if doc == 100:
        break

    # if d['heads'][0] < 0:
    #     print(i)
    # i += 1

# pbar = tqdm.tqdm(total=int(800_000_000/1024))
# i = 0
# for d in data:
#     assert len(d['input_ids']) == len(d['tok2word']) == len(d['heads']) == 1024
#     if 836 <= i <= 839:
#         print(i)
#         print(d)
#     # pbar.update()
#     i += 1
