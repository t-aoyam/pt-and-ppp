from transformers import GPT2TokenizerFast
import os, pathlib, tqdm, json, pickle, re


# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
# pkl_fp = os.path.join(DATA_DIR, 'cc100_1m_tokens.pkl')
# pkl_fp = os.path.join(DATA_DIR, 'pile_1b_tokens_gpt.pkl')
conllu_fp = os.path.join(DATA_DIR, 'cc100_1b_tokens.conllu')
# conllu_fp = os.path.join(DATA_DIR, 'pile_1b_tokens.conllu')
# conllu_fp = os.path.join(DATA_DIR, 'pile_10m_tokens.conllu')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
pad = True
# with open(pkl_fp, 'rb') as f:
#     pkl_tokens = pickle.load(f)

def open_as_generator(conllu_fp, is_first_doc=True):
    with open(conllu_fp) as f:
        for line in f:
            if is_first_doc:
                is_first_doc = False
                continue
            yield line.strip()

# idx = 0
# consecutive_errors = 0
pbar = tqdm.tqdm(total=1_000_000_000)
# pbar = tqdm.tqdm(total=1_000_000)
words, words_used, wids, heads, toks, curr_toks, tok2word = [], [], [], [], [], [], []
ctx_size = 1024
keep_one = False
# batch-specific id for conllu
out_fp = os.path.join(DATA_DIR, 'cc100_1b_tokens_heads.jsonl')
# out_fp = os.path.join(DATA_DIR, 'pile_1b_tokens_heads.jsonl')
i = 0
with open(out_fp, 'w') as f:
    for line in open_as_generator(conllu_fp):
        # if i < 700_000:
        #     i += 1
        #     continue

        # print(line)
        if not line:
            continue
        elif line.startswith('# newdoc'):
            toks.append(tokenizer.eos_token_id)
            words.append(tokenizer.eos_token)
            wids.append(-1)  # to be removed at training time
            heads.append(-1)  # to be removed at training time
        elif line.startswith('# newsent'):
            continue
        else:  # token
            wid, word, head, rel = line.split('\t')
            words.append(word)
            wids.append(int(wid)-1)  # to 0-index
            heads.append(int(head)-1)  # to 0-index
            toks.extend(tokenizer(word)['input_ids'])
        if len(toks) < ctx_size:
            continue

        word_id = 0
        num_toks = 0
        for i, tok in enumerate(toks[:ctx_size]):
            num_toks += 1
            curr_toks.append(tok)
            # print(tokenizer.decode(curr_toks).strip(), words[word_id].strip(), len(words[word_id]))
            if tokenizer.decode(curr_toks).strip() == words[word_id].strip():
                tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                for tok_id in tok_ids:
                    if tok_id < 0:  # boundary situation can have negative tok_id
                        continue
                    tok2word.append(word_id)
                word_id += 1
                curr_toks = []
            pbar.update()
            if len(curr_toks) > 1000:  # in the pretokenization script, token of length > 150 is removed
                # print(tokenizer.decode(curr_toks).strip(), words[word_id])
                raise IOError

        if curr_toks:  # if the last token in batch NOT at the word boundary; e.g. [..., ddmin, nea][polis, ...]
            # print(f'LEFTOVER CURRTOKS: {tokenizer.decode(curr_toks)}')
            tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
            for tok_id in tok_ids:
                if tok_id < 0:  # boundary situation can have negative tok_id
                    continue
                tok2word.append(word_id)
            keep_one = True
            # NO curr_toks = []
            # NO word_id += 1
        else:
            keep_one = False

        # get this batche's head and word boundary info
        # batch_wids = list(range(1, word_id+1))
        batch_wids = list(range(word_id))
        offsets = [batch_level - orig for orig, batch_level in zip(wids[:word_id], batch_wids)]
        batch_heads = [head + offset for head, offset in zip(heads[:word_id], offsets)]
        # print(len(tok2word))
        # print(tok2word[:100])
        assert len(tok2word) == ctx_size
        # print(len(batch_wids), len(batch_heads), word_id)
        assert len(batch_wids) == len(batch_heads) == word_id

        if pad:
            batch_heads = batch_heads + [-1]*(ctx_size-len(batch_heads))
        batch = {'input_ids': toks[:ctx_size],
                 'tok2word': tok2word,
                 'heads': batch_heads}
        f.write(json.dumps(batch)+'\n')

        if keep_one:
            words, wids, heads = words[-1:], wids[-1:], heads[-1:]
        else:
            words, wids, heads = [], [], []

        toks = toks[ctx_size:]
        tok2word = []

"""
    if len(words) > ctx_size:
        text = ' '.join(words)
        text = re.sub(r" <\|endoftext\|> ", r"<|endoftext|>", text)
        print(text)
        toks.extend(tokenizer(text)['input_ids'])
        words_used.extend(words)
        words = []
        word_id = 0
        num_toks = 0
        tok2word = []
        for i, tok in enumerate(toks[:ctx_size]):
            num_toks += 1
            curr_toks.append(tok)
            print(tokenizer.decode(curr_toks).strip(), words_used[word_id])
            if tokenizer.decode(curr_toks).strip() == words_used[word_id]:
                tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                for tok_id in tok_ids:
                    if tok_id < 0:  # boundary situation can have negative tok_id
                        continue
                    tok2word.append(word_id)
                word_id += 1
                curr_toks = []
            pbar.update()
            if len(curr_toks) > 10:
                raise IOError(f'text: {text}\ntokens: {curr_toks}')
        print('|||||BATCH BOUNDARY|||||')
        if curr_toks:  # if the last token in batch NOT at the word boundary; e.g. [..., ddmin, nea][police, ...]
            print(f'LEFTOVER CURRTOKS: {tokenizer.decode(curr_toks)}')
            tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
            for tok_id in tok_ids:
                if tok_id < 0:  # boundary situation can have negative tok_id
                    continue
                tok2word[tok_id] = word_id
            # NO curr_toks = []
            # NO word_id += 1
            start_id = word_id
        else:
            start_id = word_id

        batch_wids = list(range(1, word_id+1))
        offsets = [batch_level - orig for orig, batch_level in zip(wids[:word_id], batch_wids)]
        batch_heads = [head + offset for head, offset in zip(heads[:word_id], offsets)]

        words, wids, heads = words[start_id:], wids[start_id:], heads[start_id:]
        wids, heads = wids[start_id:], heads[start_id:]
        toks = toks[ctx_size:]
"""