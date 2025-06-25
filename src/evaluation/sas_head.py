import os
import torch, pathlib
from tqdm import tqdm
import argparse

# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAS_PREDS_DIR = os.path.join(DATA_DIR, 'sas_preds')
OUTPUT_DIR = os.path.join(DATA_DIR, 'sas_scores')
EWT_FP = os.path.join(DATA_DIR, 'ewt.txt')

# read the list of UD relations
with open(os.path.join(DATA_DIR, 'UD_rel_names.txt')) as f:
    REL_NAMES = [line.strip() for line in f.readlines()]

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# get gold EWT data
def get_wid_mapping(fp):
    with open(fp) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    return lines
def get_preds(fn):
    with open(os.path.join(SAS_PREDS_DIR, fn)) as f:
        preds = [
            [[int(idx) for idx in layer.split('-')]
                  for layer in line.strip().split('\t')]
                 for line in f.readlines()
        ]
    return preds

def get_by_head_acc(preds, did2gid, gid2info):
    pbar = tqdm(total=len(preds))
    results = []
    for wid, word_preds in enumerate(preds):
        gold_head, rel = gid2info[wid][3], gid2info[wid][4]
        if '--' in gold_head or gold_head == '<|endoftext|>':
            # gold_head, rel = '', ''
            pbar.update()
            continue
        gold_head = did2gid[gold_head]
        results.append([int(gold_head == head_pred) for layer_preds in word_preds for head_pred in layer_preds])
        pbar.update()
    pbar.close()
    results = torch.tensor(results).permute(1, 0)  # (head, word)
    return results

def get_per_relation_acc(by_head_results, parent_rels):
    by_rel_results = dict()
    uas = 0
    total = 0
    for rel in tqdm(REL_NAMES):
        is_this_rel = torch.tensor([int(parent_rel==rel) for parent_rel in parent_rels])
        rel_total = int(torch.sum(is_this_rel))
        is_this_rel = is_this_rel.expand(*by_head_results.shape)
        rel_correct = by_head_results * is_this_rel
        _sum = torch.sum(rel_correct, dim=1)
        by_rel_results[rel] = _sum/rel_total
        uas += float(max(_sum))
        total += rel_total
    by_rel_results['total'] = torch.sum(by_head_results, dim=1)/by_head_results.shape[1]
    uas /= total
    return uas, by_rel_results

def id2name(idx, num_head):
    lid = str(idx//num_head)  # layer ID
    hid = str(idx%num_head)  # head ID
    return '-'.join([lid, hid])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default=None,
                        help='model name whose SAS head scores will be computed. default=None')
    args = parser.parse_args()

    if args.model_name:
        fns = [fn for fn in os.listdir(SAS_PREDS_DIR) if args.model_name in fn]
    else:
        fns = [fn for fn in os.listdir(SAS_PREDS_DIR)]
    for fn in fns:
        print(fn)

        gid2info = get_wid_mapping(EWT_FP)
        did2gid = {line[0]: i for i, line in enumerate(gid2info)}

        # fn = 'EleutherAI-pythia-70m-step128-sas_preds@scaled.tsv'
        fn.split('@')
        # get prediction data
        preds = get_preds(fn)
        by_head_results = get_by_head_acc(preds, did2gid, gid2info)

        # remove EOS and roots
        preds = [pred for i, pred in enumerate(preds) if '--' not in gid2info[i][3] and gid2info[i][3] != '<|endoftext|>']
        did2gid = {line[0]: i for i, line in enumerate(gid2info)}
        gid2info = [line for line in gid2info if '--' not in line[3] and line[3] != '<|endoftext|>'][:len(preds)]
        parent_rels = [line[4] for line in gid2info]
        num_head = torch.tensor(preds).shape[-1]

        # record per head per relation accuracy, as well as overall (for extra metric)
        uas, by_rel_results = get_per_relation_acc(by_head_results, parent_rels)

        # write UAS
        uas_fn = fn.replace('sas_preds', 'sas_uas')
        with open(os.path.join(OUTPUT_DIR, uas_fn), 'w') as f:
            f.write(str(uas))

        # write each head's per relation accuracy
        out = []
        for i in tqdm(range(by_head_results.shape[0])):
            head_name = id2name(i, num_head)
            scores = [head_name] + ['-'.join([rel, str(float(by_rel_results[rel][i]))]) for rel in by_rel_results]
            out.append('\t'.join(scores))
        by_head_fn = fn.replace('sas_preds', 'sas_scores_by_head')
        with open(os.path.join(OUTPUT_DIR, by_head_fn), 'w') as f:
            f.write('\n'.join(out))

main()