"""
Author: Tatsuya Aoyama
conversion code:
dundee: 1, nsc: 2, ... (can add more)
corpus*100_000 + story*10_000 + tok_id (at most ~3000 tokens)
e.g. 112_200 corpus 1 (dundee), story 1, token 2_200

write .csv file:
["code", "word", "surprisal", "psychometric", "corpus", "model", "training", "seed", "len", "freq"]

"""
from collections import Counter
import os, pathlib, math, re, argparse, glob
import pandas as pd
from tqdm import tqdm

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
wiki_fp = r"C:\Users\aozsa\Codes\neural-networks-read-times\data\wikitext-2_train_vocab.txt"
rts_fp = os.path.join(DATA_DIR, 'rt_data', 'all_rts.tsv')

corpus2id = {'dundee': 1, 'ns': 2, 'provo': 3, 'meco': 4}

def get_freq_dct(wiki_fp):
    word_freq = Counter()
    with open(wiki_fp) as f:
        for line in f:
            token, freq = line.strip().split("\t")
            word_freq[token] = int(freq)
    return word_freq

def get_rts_dct(rts_fp):
    with open(rts_fp) as f:
        rts = [line.strip().split('\t') for line in f.readlines()]
    return {code: float(rt) for code, rt in rts}

def hatch(SURP_DIR, model_name, ablation_mode, ablation_threshold, ablation_head, word_freq, rts_dct):
    if ablation_mode:
        if ablation_threshold:
            ablation_threshold = str(int(ablation_threshold*100))
            surp_fns = [surp_fn for surp_fn in os.listdir(SURP_DIR) if \
                        surp_fn.startswith(model_name) and ablation_mode in surp_fn and f'@{ablation_threshold}' in surp_fn]
        elif ablation_head:
            surp_fns = [f"{model_name}-1024-512-surps@{ablation_head}-{ablation_mode}.tsv"]
    else:  # no threshold, then hatch all the training steps
        surp_fps = glob.glob(os.path.join(SURP_DIR, f"{model_name}*-1024-512-surps.tsv"))
        surp_fns = [fp.split(os.path.sep)[-1] for fp in surp_fps]

        # else:  # if no threshold, assume by-head ablation
            # surp_fns = [surp_fn for surp_fn in os.listdir(SURP_DIR) if \
            #             surp_fn.startswith(model_name) and ablation_mode in surp_fn and\
            #             re.search(r"surps@\d+-\d+", surp_fn)]

    output = []
    for surp_fn in tqdm(surp_fns):  # go through all surps,
        with open(os.path.join(SURP_DIR, surp_fn)) as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            code, word, surp, positions = line
            if code in rts_dct:
                rt = rts_dct[code]
            else:
                print(f"skipping: {code}")
                continue
            corpus, story_idx, word_idx = code.split('-')
            freq = math.log(word_freq[word.lower()]+1)  # natural log; +1 to make 0 freq also 0 in log space
            int_code = corpus2id[corpus]*100_000+int(story_idx)*10_000+int(word_idx)
            output.append([int_code, word, surp, rt, corpus, surp_fn.split('.')[0], 'cc100', 42, len(word), freq, positions])
    df = pd.DataFrame(output)
    df.columns = ["code", "word", "surprisal", "psychometric", "corpus", "model", "training", "seed", "len", "freq", "position"]
    if ablation_mode:
        if ablation_threshold:
            output_fn = f'harmonized_results_{model_name}@{ablation_threshold}-{ablation_mode}.csv'
            df.to_csv(os.path.join(DATA_DIR,
                                   'rt_data',
                                   output_fn
                                   ))
        elif ablation_head:
            output_fn = f'harmonized_results_{model_name}@{ablation_head}-{ablation_mode}.csv'
            df.to_csv(os.path.join(DATA_DIR,
                                   'by_head_ablation',
                                   'rt_data',
                                   output_fn
                                   ))
    else:
        output_fn = f'harmonized_results_{model_name}.csv'
        df.to_csv(os.path.join(DATA_DIR,
                               'rt_data',
                               output_fn
                               ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                        help='name of the model')
    parser.add_argument('-at', '--ablation_threshold', type=float, default=None,
                        help="induction heads with prefix matching score above this threshold will be ablated,\
                        default=1.0 (no ablation)")
    parser.add_argument('-am', '--ablation_mode', choices=['full', 'pp'], default=None,
                        help="type of ablation to perform: ['full', 'pp'], default=None")
    parser.add_argument('-ah', '--ablation_head', default=None,
                        help="head to ablate e.g. '-ah 0-3',  default=None")
    args = parser.parse_args()
    model_name, ablation_threshold, ablation_mode, ablation_head = \
        args.model_name, args.ablation_threshold, args.ablation_mode, args.ablation_head
    # model_name = 'EleutherAI-pythia-70m'
    # model_name = 'gpt2-mlp-2-layers-checkpoint-15625'
    # model_name = 'EleutherAI-pythia-70m-step1000'
    # ablation_mode = 'full'
    # ablation_threshold = None
    if ablation_threshold:
        SURP_DIR = os.path.join(DATA_DIR, 'surps')
    elif ablation_head:
        SURP_DIR = os.path.join(DATA_DIR, 'by_head_ablation', 'surps')
    else:
        SURP_DIR = os.path.join(DATA_DIR, 'surps')

    # model_name = 'gpt2-medium'
    freq_dct = get_freq_dct(wiki_fp)
    rts_dct = get_rts_dct(rts_fp)
    hatch(SURP_DIR, model_name, ablation_mode, ablation_threshold, ablation_head, freq_dct, rts_dct)

if __name__ == "__main__":
    main()