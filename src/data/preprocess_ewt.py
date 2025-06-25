"""
Take ETW train, dev, and test splits, and combine them to one text file,
where all token IDs are unique.
IDs are re-numbered from 1-index to 0-index
"""
import os
from tqdm import tqdm
from collections import Counter
import math

# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EWT_DIR = os.path.join(DATA_DIR, 'UD_English-EWT')
wiki_fp = r"C:\Users\aozsa\Codes\neural-networks-read-times\data\wikitext-2_train_vocab.txt"
add_eos = True
eos = '<|endoftext|>'
fns = ['en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu']

def get_freq_dct(wiki_fp):
    word_freq = Counter()
    with open(wiki_fp) as f:
        for line in f:
            token, freq = line.strip().split("\t")
            word_freq[token] = int(freq)
    return word_freq

word_freq = get_freq_dct(wiki_fp)
did = -1  # first doc will be 0
d_tid = 0
tid = 0
sent, rows = [], []
for fn in fns:
    with open(os.path.join(EWT_DIR, fn)) as f:
        lines = [line.strip() for line in f.readlines()]
    for line in tqdm(lines):
        if not line:
            continue
        elif line.startswith('# newdoc'):
            if sent:
                for row in sent:
                    if '--' in row[3]:
                        continue
                    pid = int(row[3].split('-')[1])
                    # print(pid, row)
                    # print(sent)
                    sent[pid][5].append(f"{row[0]}:{row[4]}")
                rows.extend(sent)
                if add_eos:
                    rows.append([f"{str(did)}-{str(d_tid+tid)}", eos, eos,
                                 eos, eos, '', 0])
                sent = []
            did += 1
            tid = 0  # reset cumulative document level token ID
            d_tid = 0  # reset document level token ID
        elif line.startswith('# sent'):
            d_tid += tid  # continue enumerating from the last sentence
        elif not line.startswith('#'):  # actual token
            tid, word, lemma, xpos, upos, morph, parent, deprel, _, _ = line.split('\t')
            if not tid.isnumeric():  # skip ellipses, etc.
                continue
            tid = int(tid)
            parent = int(parent)
            freq = math.log(word_freq[word.lower()]+1)  # natural log; +1 to make 0 freq also 0 in log space
            sent.append([f"{str(did)}-{str(d_tid+tid-1)}", word, upos,
                         f"{str(did)}-{str(d_tid+parent-1)}", deprel, [], freq])
    if sent:
        for row in sent:
            if '--' in row[3]:
                continue
            pid = int(row[3].split('-')[1])
            # print(pid, row)
            # print(sent)
            sent[pid][5].append(f"{row[0]}:{row[4]}")
        rows.extend(sent)
        if add_eos:
            rows.append([f"{str(did)}-{str(d_tid + tid)}", eos, eos,
                         eos, eos, '', 0])
        sent = []
        did += 1
        tid = 0  # reset cumulative document level token ID
        d_tid = 0  # reset document level token ID

for line in rows:
    line[5] = ','.join(line[5])
    for i, _ in enumerate(line):
        line[i] = str(line[i])

with open(os.path.join(DATA_DIR, 'ewt.txt'), 'w') as f:
    f.write('\n'.join(['\t'.join(row) for row in rows]))