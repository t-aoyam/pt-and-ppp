"""
Author: Tatsuya

*ALL 1-INDEXED*
Take reading time corpora and assign unique code to each.
Codes strictly follow corpus-storyID-tokenID, so it can be reconstructed from other data.
Writes 1 .tsv:
all_toks.tsv: each row contains a code-word pair.
all_rts.tsv: each row contains a code-rt pair.
---
corpus-storyID-tokenID    word1
dundee-1-1  Abc
dundee-1-2  def
...
ns-10-1000  xyz
---
"""
import os, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
RT_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'rt_data')

nsc_fp = os.path.join(RT_DATA_DIR, 'all_stories.tok')
nsc_rts_fp = os.path.join(RT_DATA_DIR, 'processed_RTs.tsv')
dundee_dir = os.path.join(RT_DATA_DIR, 'dundee')
provo_rts_fp = os.path.join(RT_DATA_DIR, 'Provo_Corpus-Eyetracking_Data.csv')
provo_fp = os.path.join(RT_DATA_DIR, 'Provo_Corpus-Predictability_Norms.csv')
# meco_fp = os.path.join(RT_DATA_DIR, 'joint_data_trimmed.csv')
# fixed MECO below
meco_fp = os.path.join(RT_DATA_DIR, 'MECO_fixed_en.csv')
output_dir = RT_DATA_DIR

def get_nsc_stories(nsc_fp):
    # all stories in 1 file
    with open(nsc_fp) as f:
        lines = [line.strip().split('\t') for line in f.readlines()[1:] if line.strip()]
    toks = []
    codes = []
    for line in lines:
        tok, idx, story_idx = line
        code = '-'.join(['ns', story_idx, idx])
        toks.append(tok)
        codes.append(code)
    return toks, codes

def get_nsc_rts(nsc_rts_fp):
    with open(nsc_rts_fp) as f:
        lines = f.readlines()[1:]
    rts = []
    prev_code = None
    for line in tqdm(lines):
        fields = line.split('\t')
        story_idx, token_idx, word, avg_rt = fields[3], fields[4], fields[6], fields[8]
        code = '-'.join(['ns', str(int(story_idx)), str(int(token_idx))])
        if code == prev_code:
            continue
        rts.append([code, word, avg_rt])
        prev_code = code
    return rts

def get_dundee_stories(dundee_dir):
    # 1 story per file
    codes, toks, rts = [], [], []
    for fp in os.listdir(dundee_dir):
        if not fp.endswith('txt') or 'avg' not in fp:  # only look at "avg" for now
            continue
        story_idx = str(int(fp[2:4]))  # normalize 01 to 1, etc.

        with open(os.path.join(dundee_dir, fp)) as f:
            lines = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        for idx, line in enumerate(lines):
            tok, rt = line
            code = '-'.join(['dundee', story_idx, str(idx+1)])
            toks.append(tok)
            codes.append(code)
            rts.append(rt)
    return toks, codes, rts

def get_provo_stories(provo_fp, provo_rts_fp):
    df = pd.read_csv(provo_fp, encoding='latin-1')
    df['Text'] = df['Text'].str.replace(r"Õ", "'", regex=False)
    texts = df['Text'].unique().tolist()

    df = pd.read_csv(provo_rts_fp, encoding='latin-1')
    df['Word'] = df['Word'].str.replace(r"Õ", "'", regex=False)
    df = df.drop_duplicates('Word_Unique_ID', keep='first')

    # first create idx: word dict
    # go through row by row and fix (1) missing period and (2) missing first word
    idx2word = dict()
    prev_tid = 0  # dummy first text ID
    prev_wid = 0  # dummy first word ID
    for i in range(len(df)):
        row = df.iloc[i,]
        tid, wid, word = row[['Text_ID', 'Word_Number', 'Word']].to_list()
        if pd.isna(wid):
            continue
        tid, wid = int(tid), int(wid)
        # if wid != prev_wid + 1:
        #     print(tid, wid-1, word)
        idx = '-'.join(['provo', str(tid), str(wid)])
        if tid != prev_tid and wid != 1:  # new text and missing first word
            # get the first word from texts
            missing_idx = '-'.join(['provo', str(tid), '1'])
            missing_word = texts[tid-1].split()[0]
            print(f"adding the missing word at {missing_idx}: {missing_word}")
            idx2word[missing_idx] = missing_word
        if i == len(df)-1 or tid != int(df.iloc[i+1, ]['Text_ID']):  # if last word of a text
            if word[-1] not in ['.', ',', '!', '?', '"']:
                print(f"adding '.' to {idx}: {word}")
                word += '.'
        idx2word[idx] = word
        prev_tid = tid
        # prev_wid = wid

    # add missed cases
    idx2word['provo-55-3'] = 'probably'
    idx2word['provo-55-10'] = 'a'

    # then create idx: [rts] dict  (missing items will NOT be added here)
    provo_rt_var = 'IA_FIRST_RUN_DWELL_TIME'
    df = pd.read_csv(provo_rts_fp, encoding='latin-1')  # add duplicates in (different subjects)
    df['Word'] = df['Word'].str.replace(r"Õ", "'", regex=False)
    lists = list(df[['Text_ID', 'Word_Number', 'Word', provo_rt_var]].to_dict(orient='list').values())
    idx2rt = dict()
    for tid, wid, word, rt in zip(*lists):
        if pd.isna(wid) or pd.isna(rt):
            continue
        tid, wid, rt = int(tid), int(wid), float(rt)
        idx = '-'.join(['provo', str(tid), str(wid)])
        if idx != 'provo-18-3':  # known bug
            # print(idx2word[idx].strip('.,!?"'), word.strip('.,!?"'))
            assert idx2word[idx].strip('.,!?"') == word.strip('.,!?"')
        idx2rt[idx] = idx2rt.get(idx, []) + [rt]
    for key in idx2rt:
        idx2rt[key] = float(np.mean(idx2rt[key]))

    toks = [[code, word] for code, word in idx2word.items()]
    rts = [[code, rts] for code, rts in idx2rt.items()]
    toks.sort(key=lambda x: int(x[0].split('-')[2]))
    toks.sort(key=lambda x: int(x[0].split('-')[1]))
    rts.sort(key=lambda x: int(x[0].split('-')[2]))
    rts.sort(key=lambda x: int(x[0].split('-')[1]))

    return toks, rts

def get_meco_stories(meco_fp):
    # first get idx: word dict
    meco_rt_var = 'firstrun.dur'
    df = pd.read_csv(meco_fp)
    df = df[df['lang']=='en']
    lists = list(df[['itemid', 'ianum']].to_dict(orient='list').values())
    words = df['ia'].to_list()
    idxs = ['-'.join(['meco', str(int(tid)), str(int(wid))]) for tid, wid in zip(*lists)]
    idx2word = {idx:word for idx, word in zip(idxs, words) if not pd.isna(word)}
    df['idx'] = idxs
    rts = df[meco_rt_var].to_list()
    idx2rt = dict()
    for idx, rt in zip(idxs, rts):
        if pd.isna(rt) or idx not in idx2word:
            continue
        idx2rt[idx] = idx2rt.get(idx, []) + [float(rt)]
    for idx in idx2rt:
        idx2rt[idx] = float(np.mean(idx2rt[idx]))

    toks = [[code, word] for code, word in idx2word.items()]
    rts = [[code, rts] for code, rts in idx2rt.items()]
    toks.sort(key=lambda x: int(x[0].split('-')[2]))
    toks.sort(key=lambda x: int(x[0].split('-')[1]))
    rts.sort(key=lambda x: int(x[0].split('-')[2]))
    rts.sort(key=lambda x: int(x[0].split('-')[1]))

    return toks, rts

dtoks, dcodes, drts = get_dundee_stories(dundee_dir)
ntoks, ncodes = get_nsc_stories(nsc_fp)
nrts = get_nsc_rts(nsc_rts_fp)
assert [[code, word] for code, word, _ in nrts] == [[code, word] for code, word in zip(ncodes, ntoks)],\
    ValueError('Tokens do not match')
# TODO: if the first word of each story is discarded, put a dummy first word back in?
ptoks, prts = get_provo_stories(provo_fp, provo_rts_fp)
mtoks, mrts = get_meco_stories(meco_fp)

# TODO: more corpora?

# write toks
toks = ntoks + dtoks
codes = ncodes + dcodes
out = ['\t'.join([code, tok]) for code, tok in zip(codes, toks)] +\
       ['\t'.join(item) for item in ptoks] +\
       ['\t'.join(item) for item in mtoks]

with open(os.path.join(output_dir, 'all_toks.tsv'), 'w') as f:
    f.write('\n'.join(out))

# write rts
rts = nrts + [[c, w, t] for c, w, t in zip(dcodes, dtoks, drts)]
out = ['\t'.join([code, rt]) for code, rt in zip(codes, [r for _, _, r in rts])] +\
    ['\t'.join([code, str(rt)]) for code, rt in prts] +\
    ['\t'.join([code, str(rt)]) for code, rt in mrts]

with open(os.path.join(output_dir, 'all_rts.tsv'), 'w') as f:
    f.write('\n'.join(out))