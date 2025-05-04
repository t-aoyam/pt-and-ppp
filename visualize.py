"""
custom GPT2 checkpoints are:
["0.5M", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M",
 "0.5B", "1.0B", "1.5B", "2.0B", "2.5B", "3.0B", "3.5B", "4.0B", "4.5B", "5.0B",
 "5.5B", "6.0B", "6.5B", "7.0B", "7.5B", "8.0B", "8.5B", "9.0B", "9.5B", "10.0B" ]

Selected 15 Pythia checkpoints are:
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]
and these correspond to:
[2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1B, 2B, 4B, 6B, 8B, 10B]
and in terms of the indices, these correspond to:
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 21, 25, 29]
"""
import os, pathlib
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple

corpora = ['dundee', 'meco', 'provo', 'ns']
# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LL_DIR = os.path.join(DATA_DIR, 'analysis_checkpoints')
LOSS_DIR = os.path.join(DATA_DIR, 'losses')
PMS_DIR = os.path.join(DATA_DIR, 'induction_heads')
def dll_steps(token_counts, dll, corpus, xlabel):
    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.plot(token_counts, dll)
    ax.set_xticklabels(token_counts, rotation=60, rotation_mode='anchor', ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('delta log-likelihood')
    ax.set_title(corpus)
    plt.show()

def get_dlls(csv_fn, step_id=5):
    dll = pd.read_csv(os.path.join(LL_DIR, csv_fn), index_col=0)
    results = dict()
    for corpus in corpora:
        for i in dll.loc[dll['corpus'] == corpus].index:
            i -= 1
            steps = dll.iloc[i,]['model'].split('-')[step_id].strip('step')
            results[corpus] = results.get(corpus, []) + [[int(steps), float(dll.iloc[i,]['delta_test_mean'])]]
    for corpus in corpora:
        results[corpus].sort(key=lambda x: x[0])
    return results

def ppl2ppp(checkpoint2loss, dll, title):
    plt.scatter([item[1] for item in checkpoint2loss], [item[1] for item in dll])
    plt.grid(True)
    plt.title(title)
    plt.xlabel('val perplexity')
    plt.ylabel('delta log-likelihood')
    plt.show()

def train2icl(checkpoint2loss, token_counts, title):
    plt.plot(token_counts, [item[2]for item in checkpoint2loss])
    plt.grid(True)
    plt.xticks(rotation=60)
    plt.title(title)
    plt.xlabel('# training tokens')
    plt.ylabel('ICL score')
    plt.show()

def train2ppl(checkpoint2loss, token_counts, title):
    plt.plot(token_counts, [item[1]for item in checkpoint2loss])
    plt.grid(True)
    plt.xticks(rotation=60)
    plt.title(title)
    plt.xlabel('# training tokens')
    plt.ylabel('perplexity')
    # plt.ylim((250,300))
    plt.show()

def get_checkpoint2loss(model_name, loss_dir, ablation_mode=None, ablation_threshold=None, step_id=5):
    loss_fns = os.listdir(loss_dir)
    checkpoint2loss = []
    for fn in loss_fns:
        if model_name not in fn or 'losses' in fn:
            continue
        if not ablation_mode:
            if '@' in fn:
                continue
        if ablation_mode:
            if f'@{str(int(ablation_threshold * 100))}-{ablation_mode}' not in fn:
                continue
        with open(os.path.join(LOSS_DIR, fn)) as f:
            ppl, icl, ppl_sd, icl_sd = f.readlines()[0].strip().split('\t')
        c = fn.split('-')[step_id].strip('step')
        checkpoint2loss.append([int(c), float(ppl), float(icl)])
    checkpoint2loss.sort(key=lambda x:x[0])

    return checkpoint2loss

def dll_steps_all(token_counts: List[str], dlls: List[Dict[str, List[int]]],
                  labels: List[str], colors, icl_bump_ids, highlight=True, families=''):
    """

    :param token_counts:
    :param dlls: [{corpus: [step, dll], [step, dll], ...}, {...}]
    :param labels:
    :param colors:
    :param icl_bump_ids:
    :param highlight:
    :return:
    """
    # assert len(dundees) == len(nss) == len(labels)
    fig, axs = plt.subplots(max(1, len(families)), len(dlls[0]),
                            figsize=(4 * len(dlls[0]), 4* max(1, len(families))))
    if not families:
        axs = [axs]
    for row in axs:
        for ax in row:
            ax.grid(True)
    for row, family in enumerate(families):
        model_ids = [labels.index(l) for l in labels if family in l]
        for model_id in model_ids:
            model = dlls[model_id]
            for corpus_id, corpus in enumerate(list(model.keys())):
                if '@' in labels[model_id]:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                if len(model[corpus]) == 30:
                    axs[row][corpus_id].plot(token_counts, [item[1] for item in model[corpus]],
                                        color=colors[model_id],
                                        linestyle=linestyle, label=labels[model_id])
                    bump_id = icl_bump_ids[model_id]
                else:
                    axs[row][corpus_id].plot(indices, [item[1] for item in model[corpus]],
                                        color=colors[model_id],
                                        linestyle=linestyle, label=labels[model_id])
                    bump_id = indices[icl_bump_ids[model_id]]
                axs[row][corpus_id].set_title(corpus)
                axs[row][corpus_id].set_xticks(range(len(token_counts)), rotation=60, size=6)
                axs[row][corpus_id].set_xticklabels(token_counts, rotation=60, size=6)
                if highlight and icl_bump_ids[model_id]:
                    axs[row][corpus_id].axvline(x=bump_id, color=colors[model_id], linestyle='-',
                                           linewidth=10, alpha=0.2, label=None)
        axs[row][-1].legend()
    fig.suptitle('DLL vs training steps')
    fig.supylabel('DLL')
    plt.show()

def icl_steps_all(token_counts: List[str], checkpoint2loss: List[int],
                  labels: List[str], colors, highlight=True):

    assert len(checkpoint2loss) == len(labels)
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    ax.grid(True)
    for i, c in enumerate(checkpoint2loss):
        if '@' in labels[i]:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        icls = [item[2] for item in c]
        bump = max(0, get_sudden_rise(icls))
        if len(c) == 30:
            ax.plot(range(len(token_counts)), icls, color=colors[i], linestyle=linestyle, label=labels[i])
            bump_id = bump
        else:
            ax.plot(indices, [item[2] for item in c], color=colors[i], linestyle=linestyle, label=labels[i])
            bump_id = indices[bump]
        if highlight and bump:
            ax.axvline(x=bump_id, color=colors[i], linestyle='-', linewidth=5, alpha=0.2, label=None)
    ax.set_xticks(range(len(token_counts)))
    ax.set_xticklabels(token_counts, rotation=60, size=6)

    fig.suptitle('ICL vs training steps')
    fig.supylabel('ICL')
    fig.legend()
    plt.show()

def get_sudden_rise(icl_scores, threshold=0.2):
    deltas = [0] + [icl_scores[i]-icl_scores[i-1] > threshold for i in range(1, len(icl_scores))]
    if True in deltas:
        return deltas.index(True)  # first bump larger than threshold, otherwise -1
    return 0

def get_colors_from_cmap(cmap_name, k):
    cmap = plt.get_cmap(cmap_name)  # Get the colormap
    return [cmap(i / (k - 1)) for i in range(k)]  # Sample k colors equally spaced

def get_checkpoint2pms(model_name, pms_dir, ablation_mode=None, ablation_threshold=None, step_id=5):
    pms_fns = os.listdir(pms_dir)
    checkpoint2pms = dict()
    for fn in pms_fns:
        if model_name not in fn:
            continue
        if not ablation_mode:
            if '@' in fn:
                continue
        if ablation_mode:
            if f'@{str(int(ablation_threshold * 100))}-{ablation_mode}' not in fn:
                continue
        c = int(fn.split('-')[step_id].strip('step'))
        with open(os.path.join(PMS_DIR, fn)) as f:
            heads = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        for head_idx, score in heads:
            l, h = [int(item) for item in head_idx.split('-')]
            checkpoint2pms[c] = checkpoint2pms.get(c, []) + [(l, h, float(score))]
        checkpoint2pms[c].sort(key=lambda x:x[1])
        checkpoint2pms[c].sort(key=lambda x:x[0])

    return checkpoint2pms

def pms_steps_all(token_counts: List[str], checkpoint2pmss: List[Dict[int, List[Tuple]]],
                  labels: List[str], colors, highlight=True):

    assert len(checkpoint2pmss) == len(labels)
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    ax.grid(True)
    for i, model in enumerate(checkpoint2pmss):
        # convert {checkpoint: [(l, h, score), ...], checkpoint: [(l, h, score), ...]} to
        # [checkpoint, max_score]
        checkpoint2max_pms = [(c, max([item[2] for item in checkpoint2pmss[i][c]]))
                              for c in sorted(list(checkpoint2pmss[i].keys()))]
        if '@' in labels[i]:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        max_pmss = [item[1] for item in checkpoint2max_pms]
        bump = max(0, get_sudden_rise(max_pmss))
        if len(model) == 30:
            ax.plot(range(len(token_counts)), max_pmss, color=colors[i], linestyle=linestyle, label=labels[i])
            bump_id = bump
        else:
            ax.plot(indices, max_pmss, color=colors[i], linestyle=linestyle, label=labels[i])
            bump_id = indices[bump]
        if highlight and bump:
            ax.axvline(x=bump_id, color=colors[i], linestyle='-', linewidth=5, alpha=0.2, label=None)
    ax.set_xticks(range(len(token_counts)))
    ax.set_xticklabels(token_counts, rotation=60, size=6)

    fig.suptitle('Max prefix matching score vs training steps')
    fig.supylabel('Max prefix matching score')
    fig.legend(loc='center right')
    plt.show()

token_counts = ["0.5M", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M"]\
    + [f'{i / 10}B' for i in range(5, 101, 5)]

indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 21, 25, 29]

"""All MODELS"""
models = [
    ('gpt2-mlp-0-layers', None, None),
    ('gpt2-mlp-1-layers', None, None),
    ('gpt2-mlp-2-layers', None, None),
    ('EleutherAI-pythia-70m', None, None),
    ('EleutherAI-pythia-160m', None, None),
    ('EleutherAI-pythia-410m', None, None),
    # ('gpt2-mlp-2-layers', 0.6, 'full'),
    # ('gpt2-mlp-2-layers', 0.1, 'full'),
    # ('gpt2-mlp-2-layers', 0.1, 'pp'),
    # ('EleutherAI-pythia-70m', None, None),
    # ('EleutherAI-pythia-70m', 0.9, 'full'),
    # ('EleutherAI-pythia-70m', 0.2, 'full'),
    # ('EleutherAI-pythia-70m', 0.1, 'full'),
    # ('EleutherAI-pythia-160m', None, None),
    # ('EleutherAI-pythia-410m', None, None),
    # ('EleutherAI-pythia-410m', 0.1, 'full'),
]

model_names, unique_models, dlls, losses, pmss = [], [], [], [], []

for info in models:
    model, ablation_threshold, ablation_mode = info
    if ablation_mode:
        model_name = f'{model}@{str(int(ablation_threshold*100))}-{ablation_mode}'
    else:
        model_name = model
    delta_csv = f'model_deltas_{model_name}.csv'
    model_names.append(model_name)
    step_id = 5 if 'gpt' in model else 3

    dll = get_dlls(delta_csv, step_id=step_id)
    dlls.append(dll)

    loss_icl = get_checkpoint2loss(model, LOSS_DIR, ablation_mode,
                                   ablation_threshold, step_id=step_id)
    losses.append(loss_icl)

    if model_name.split('@')[0] not in unique_models and '0-layer' not in model_name:
        unique_models.append(model_name.split('@')[0])
        pms = get_checkpoint2pms(model, PMS_DIR, ablation_mode,
                                 ablation_threshold, step_id=step_id)
        pmss.append(pms)

for l in losses:
    print(len(l))

icl_bump_ids = [get_sudden_rise([item[2] for item in c], threshold=0.15)\
                for c in losses]
losses[2]
# icl_bump_ids.append(0)

families = ['gpt', 'pythia']
dll_steps_all(token_counts, dlls, model_names,
              colors=get_colors_from_cmap('viridis', len(model_names)),
              families=families, icl_bump_ids=icl_bump_ids)

icl_steps_all(token_counts, losses, model_names, get_colors_from_cmap('viridis', len(model_names)))

pms_steps_all(token_counts, pmss, unique_models, get_colors_from_cmap('viridis', len(unique_models)))



"""0 LAYER MODEL"""

dundee_0, ns_0 = get_dlls('model_deltas_0_layers.csv')

# training steps vs dll
dll_steps(token_counts, [item[1] for item in dundee_0], 'dundee', '# training tokens')
dll_steps(token_counts, [item[1] for item in ns_0], 'natural stories', '# training tokens')
# loss vs dll
model_name = 'gpt2-mlp-0-layers'
checkpoint2loss_0 = get_checkpoint2loss(model_name, LOSS_DIR)
ppl2ppp(checkpoint2loss_0, dundee_0, '')
ppl2ppp(checkpoint2loss_0, ns_0, '')
train2icl(checkpoint2loss_0, token_counts, '0 layer model')
train2ppl(checkpoint2loss_0, token_counts, '0 layer model')

"""1 LAYER MODEL"""
dundee_1, ns_1 = get_dlls('model_deltas_1_layers.csv')

# training steps vs dll
dll_steps(token_counts, [item[1] for item in dundee_1], 'dundee', '# training tokens')
dll_steps(token_counts, [item[1] for item in ns_1], 'natural stories', '# training tokens')
# loss vs dll
model_name = 'gpt2-mlp-1-layers'
checkpoint2loss_1 = get_checkpoint2loss(model_name, LOSS_DIR)
ppl2ppp(checkpoint2loss_1, dundee_1, '')
ppl2ppp(checkpoint2loss_1, ns_1, '')
train2icl(checkpoint2loss_1, token_counts, '1 layer model')
train2ppl(checkpoint2loss_1, token_counts, '1 layer model')

"""2 LAYER MODEL"""
dundee_2, ns_2 = get_dlls('model_deltas_2_layers.csv')

# training steps vs dll
dll_steps(token_counts, [item[1] for item in dundee_2], 'dundee', '# training tokens')
dll_steps(token_counts, [item[1] for item in ns_2], 'natural stories', '# training tokens')
# loss vs dll
model_name = 'gpt2-mlp-2-layers'
checkpoint2loss_2 = get_checkpoint2loss(model_name, LOSS_DIR)
ppl2ppp(checkpoint2loss_2, dundee_2, '')
ppl2ppp(checkpoint2loss_2, ns_2, '')
train2icl(checkpoint2loss_2, token_counts, '0 layer model')
train2ppl(checkpoint2loss_2, token_counts, '0 layer model')



"""GPT2 MEDIUM"""

fn = 'model_deltas_gpt2medium.csv'
dll = pd.read_csv(os.path.join(LL_DIR, fn), index_col=0)
dundee = []
ns = []
ctx_sizes = [2**i for i in range(3, 11)]
for i in dll.loc[dll['corpus']=='dundee'].index:
    i -= 1
    ctx = dll.iloc[i,]['model'].split('-')[2]
    dundee.append([int(ctx), float(dll.iloc[i,]['delta_test_mean'])])
dundee.sort(key=lambda x:x[0])
for i in dll.loc[dll['corpus']=='ns'].index:
    i -= 1
    ctx = dll.iloc[i,]['model'].split('-')[2]
    ns.append([int(ctx), float(dll.iloc[i,]['delta_test_mean'])])
# ns.sort(key=lambda x:x[0])
# dundee

dll_steps(ctx_sizes, [item[1] for item in dundee], 'dundee', 'context size')
dll_steps(ctx_sizes, [item[1] for item in ns], 'natural stories', 'context size')