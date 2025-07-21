
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os
import pathlib, tqdm, spacy, re, json
import string
from spacy.tokens import Doc

def nlp_with_pretokenized_text(text, tokenizer, nlp):
    # Pre-tokenize the text using GPT-2
    tokens = tokenizer.tokenize(text)
    ids = tokenizer(text)['input_ids']
    words = []
    spaces = []
    word = []
    for i, tok in enumerate(tokens):
        # if the token is a continuation of the previous word and if it's not a punctuation
        if not tok.startswith('Ġ') and tok.strip() not in string.punctuation:
            # if this is a start of a new word, specify that there's no space
            if not word:
                spaces.append(False)
            word.append(ids[i])
        # if the token starts with a space (Ġ) it's a new word
        else:
            # remaining toks
            if word:
                decoded = tokenizer.decode(word).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                else:
                    spaces.pop()
                word = []
            if tok in string.punctuation:
                decoded = tokenizer.decode(ids[i]).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                    spaces.append(False)
            else:
                decoded = tokenizer.decode(ids[i]).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                    spaces.append(True)
    # last remaining tokens
    if word:
        decoded = tokenizer.decode(word).strip().strip('Ġ')
        if decoded:
            words.append(decoded)
        else:
            spaces.pop()
    spaces = spaces[1:] + [False]
    if '' in words:
        print(words, tokens)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    # Run the SpaCy pipeline on the custom Doc
    doc = nlp.get_pipe("tok2vec")(doc)
    doc = nlp.get_pipe("parser")(doc)
    return doc

def preprocess(sample):
    # text = sample['text'].strip('\n').strip()
    text = sample['text'].strip()
    if not text or max([len(word) for word in text.split()]) > 150 or len(text) > 1_000_000:
        return None
    # text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f]', ' ', text)
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f\v\x00-\x1F\x85]', ' ', text)
    text = re.sub(r' +', ' ', text)
    if not text:
        return None
    return text

def main():
    ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer"])
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    pad = True

    target_num_toks = 1_000_000_000  # should add up to 1b tokens
    out_fp = os.path.join(DATA_DIR, 'cc100_1b_tokens_heads.jsonl')
    words, words_used, wids, heads, toks, curr_toks, tok2word = [], [], [], [], [], [], []
    ctx_size = 1024

    with open(out_fp, 'w') as f:
        # get data from HF
        data = load_dataset('cc100', 'en', streaming=True)
        data = data['train']
        pbar = tqdm.tqdm(total=target_num_toks)
        num_toks = 0

        # go over each text chunk: text chunk can be a whole artcile or just a paragraph, depending on the corpus
        for sample in data:
            text = preprocess(sample)
            if not text:
                continue
            if not text.strip('\n').strip():  # CC100 marks text boundary by \n\n
                toks.append(tokenizer.eos_token_id)
                words.append(tokenizer.eos_token)
                wids.append(-1)  # to be removed at training time
                heads.append(-1)  # to be removed at training time
                continue

            nlped = nlp_with_pretokenized_text(text, tokenizer, nlp)
            tokenized = tokenizer(text, truncation=False)
            toks.extend(tokenized['input_ids'])
            offset = -1
            for sent in nlped.sents:
                for word in sent:
                    words.append(word.text)
                    wids.append(word.i-offset-1)  # 0 indexed at each sentence
                    heads.append(word.head.i-offset-1)  # 0 indexed at each sentence
                    # rels.append(word.dep_)
                offset += len(sent)

            while len(toks) > ctx_size:
                word_id = 0
                for i, tok in enumerate(toks[:ctx_size]):
                    curr_toks.append(tok)
                    if tokenizer.decode(curr_toks).strip('Ġ').strip() == words[word_id].strip('Ġ').strip():
                        tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                        for tok_id in tok_ids:
                            if tok_id < 0:  # boundary situation can have negative tok_id
                                continue
                            tok2word.append(word_id)
                        word_id += 1
                        curr_toks = []
                    if len(curr_toks) > 300:  # in the pretokenization script, token of length > 150 is removed
                        print('too many curr toks')
                        print(tokenizer.decode(curr_toks).strip(), words[word_id], toks)
                        raise IOError

                if curr_toks:  # if the last token in batch NOT at the word boundary; e.g. [..., ddmin, nea][polis, ...]
                    tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                    for tok_id in tok_ids:
                        if tok_id < 0:  # boundary situation can have negative tok_id
                            continue
                        tok2word.append(word_id)

                batch_wids = list(range(word_id))
                offsets = [batch_level - orig for orig, batch_level in zip(wids[:word_id], batch_wids)]
                batch_heads = [head + offset for head, offset in zip(heads[:word_id], offsets)]
                assert len(tok2word) == ctx_size
                assert len(batch_wids) == len(batch_heads) == word_id

                if pad:
                    batch_heads = batch_heads + [-1]*(ctx_size-len(batch_heads))
                batch = {'input_ids': toks[:ctx_size],
                         'tok2word': tok2word,
                         'heads': batch_heads}
                f.write(json.dumps(batch)+'\n')

                words, wids, heads = words[word_id:], wids[word_id:], heads[word_id:]

                toks = toks[ctx_size:]
                tok2word = []

                num_toks += ctx_size
                pbar.update(ctx_size)

            if num_toks >= target_num_toks:
                print('done')
                break