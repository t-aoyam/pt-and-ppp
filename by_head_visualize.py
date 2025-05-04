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
import copy
import os, pathlib, scipy, glob, re, tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
from scipy.stats import pearsonr

# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ABLATION_DATA_DIR = os.path.join(DATA_DIR, 'by_head_ablation')
DLL_DIR = os.path.join(ABLATION_DATA_DIR, 'analysis_checkpoints')
LOSS_DIR = os.path.join(ABLATION_DATA_DIR, 'losses')

corpora = ['dundee', 'meco', 'provo', 'ns']
corpora_labels = {'dundee': 'Dundee', 'meco': 'MECO', 'provo':'Provo', 'ns':'NS'}
proper_model_name = {'gpt': 'GPT2', 'pythia': 'Pythia'}
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
    "text.usetex": True,
    "font.size": 9,
    #    'hatch.linewidth': 0.1
})


def get_unablated(model_name, data_dir, corpora=corpora, step=None, ctx_size=1024, stride=512,
                  ) -> Dict:
    """
    :param model_name:
    :param data_dir:
    :param step:
    :param ctx_size:
    :param stride:
    :param step_id:
    :return: {step: {loss: loss, icl: icl, corpus: (corpus_dll, corpus_se), ...}
    """

    results = dict()

    # loss, icl
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    step_id = get_step_id(model_name)
    fps = glob.glob(os.path.join(data_dir, 'losses', f"{model_name}-{step_name}*-{ctx_size}-{stride}-loss.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        step = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        with open(fp) as f:
            loss, icl, loss_std, icl_std = f.readlines()[0].split('\t')
        results[step] = {'loss': float(loss), 'icl': float(icl)}

    # DLL
    fp = os.path.join(data_dir, 'analysis_checkpoints', f"model_deltas_{model_name}.csv")
    df = pd.read_csv(os.path.join(fp), index_col=0)
    for corpus in corpora:
        for i in df.loc[df['corpus'] == corpus].index:
            i -= 1
            c = int(re.sub(r'[a-z]', '', df.iloc[i,]['model'].split('-')[step_id]))
            mean, se = float(df.iloc[i,]['delta_test_mean']), float(df.iloc[i,]['delta_test_sem'])
            if c not in results: results[c] = dict()
            results[c].update({corpus: (mean, se)})

    # BLiMP
    fps = glob.glob(os.path.join(data_dir, 'blimp_scores', f"{model_name}-{step_name}*-blimp.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        step = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        with open(fp) as f:
            line = f.readlines()[0].strip()
            if len(line) > 10_000:  # if binary
                acc = sum([int(num) for num in line])/len(line)
            else:
                acc = float(line)
        results[step]['blimp'] = acc

    # UAS
    fps = glob.glob(os.path.join(data_dir, 'sas_scores', f"{model_name}-{step_name}*-sas_uas@*.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        step = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        with open(fp) as f:
            line = f.readlines()[0].strip()
            uas = float(line)
        results[step][f'uas_{"unscaled" if "unscaled" in fn else "scaled"}'] = uas

    return results

def get_best_pms_per_step(baseline, ablation):
    steps = ablation.keys()
    for step in steps:
        scores = [item[2] for item in ablation[step]['pms']]
        best = max(scores)
        baseline[step]['best_pms'] = best
    return baseline

def get_step_id(model_name):
    if 'pythia' in model_name:
        return 3
    elif 'layer' in model_name:  # old GPT name
        return 5
    elif '-s' in model_name:  # gpt-mlp-l2-b4-r3-s1-checkpoint-xxx
        return 7
    else:  # gpt-mlp-l2-b4-r3-checkpoint-xxx
        return 6
def get_head_scores(model_name, data_dir, corpora=['dundee', 'ns', 'meco', 'provo'],
                    ablation_mode='pp', ctx_size=1024, stride=512):
    step_id = get_step_id(model_name)
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    head_scores = dict()  # {step: {metric: [head, layer, score, (stderr)]}

    # 1. head-ablated DLL for each corpus
    _dir = os.path.join(data_dir, 'by_head_ablation', 'analysis_checkpoints')
    fps = glob.glob(os.path.join(_dir, f"model_deltas_{model_name}-{step_name}*@*{ablation_mode}.csv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        l, h, mode = fn.split('.')[0].split('@')[-1].split('-')  # model-name-checkpoint-loss@layer-head-mode.tsv
        if ablation_mode:
            assert mode == ablation_mode
        df = pd.read_csv(os.path.join(fp), index_col=0)
        for corpus in corpora:
            for i in df.loc[df['corpus'] == corpus].index:
                i -= 1
                c = int(re.sub(r'[a-z]', '', df.iloc[i,]['model'].split('-')[step_id]))
                dll_mean, dll_se = float(df.iloc[i,]['delta_test_mean']), float(df.iloc[i,]['delta_test_sem'])
                if c not in head_scores:
                    head_scores[c] = dict()
                head_scores[c][corpus] = head_scores[c].get(corpus, []) + \
                                            [(int(l), int(h), dll_mean, dll_se)]

    # 2. head-ablated BLiMP
    _dir = os.path.join(data_dir, 'by_head_ablation', 'blimp_scores')
    fps = glob.glob(os.path.join(_dir, f"{model_name}-{step_name}*-blimp@*{ablation_mode}.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        l, h, mode = fn.split('.')[0].split('@')[-1].split('-')  # model-name-checkpoint-loss@layer-head-mode.tsv
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        if c not in head_scores:
            head_scores[c] = dict()
        if ablation_mode:
            assert mode == ablation_mode
        with open(os.path.join(_dir, fp)) as f:
            line = f.readlines()[0].strip()
        if len(line) > 10_000:  # if binary
            acc = sum([int(num) for num in line])/len(line)
        else:
            acc = float(line)
        head_scores[c]['blimp'] = head_scores[c].get('blimp', []) + [(int(l), int(h), acc)]

    # 3. head-ablated ICL and loss
    _dir = os.path.join(data_dir, 'by_head_ablation', 'losses')
    loss_fps = glob.glob(os.path.join(_dir, f"{model_name}-{step_name}*-{ctx_size}-{stride}-loss@*-{ablation_mode}.tsv"))
    for fp in loss_fps:
        with open(fp) as f:
            ppl, icl, ppl_std, icl_std = f.readlines()[0].strip().split('\t')
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        l, h, mode = fn.split('.')[0].split('@')[-1].split('-')  # model-name-checkpoint-loss@layer-head-mode.tsv
        if ablation_mode:
            assert mode == ablation_mode
        head_scores[c]['icl'] = head_scores[c].get('icl', []) + [(int(l), int(h), float(icl))]
        head_scores[c]['loss'] = head_scores[c].get('loss', []) + [(int(l), int(h), float(ppl))]

    # 3. head SAS score
    _dir = os.path.join(data_dir, 'sas_scores')
    fps = glob.glob(os.path.join(_dir, f"{model_name}-{step_name}*-sas_scores_by_head@*.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        if c not in head_scores:
            head_scores[c] = dict()
        scaled = fn.split('.')[0].split('@')[-1]
        metric = 'sas_' + scaled
        with open(os.path.join(_dir, fp)) as f:
            lines = [line.strip() for line in f.readlines()]
        for line in lines:
            scores = line.split('\t')
            l, h = [int(item) for item in scores[0].split('-')]
            for score in scores[1:]:
                split = score.split('-')  # TODO: use something other than -
                rel, acc = split[0], float('-'.join(split[1:]))
                if metric not in head_scores[c]:
                    head_scores[c][metric] = dict()
                head_scores[c][metric][rel] = head_scores[c][metric].get(rel, []) + [(l, h, acc)]

    # 4. head PMS
    _dir = os.path.join(data_dir, 'induction_heads')
    fps = glob.glob(os.path.join(_dir, f"{model_name}-{step_name}*-induction_heads.tsv"))
    for fp in fps:
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        if c not in head_scores:
            head_scores[c] = dict()
        with open(os.path.join(_dir, fn)) as f:
            heads = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        for head_idx, score in heads:
            l, h = [int(item) for item in head_idx.split('-')]
            head_scores[c]['pms'] = head_scores[c].get('pms', []) + [(l, h, float(score))]

    for c in head_scores:
        for metric in head_scores[c]:
            if metric in ['dundee', 'meco', 'provo', 'ns', 'pms', 'blimp', 'icl', 'loss']:
                head_scores[c][metric].sort(key=lambda x:x[1])  # sort by head first
                head_scores[c][metric].sort(key=lambda x:x[0])  # sort by layer
            else:  # sas
                for rel in head_scores[c][metric]:
                    head_scores[c][metric][rel].sort(key=lambda x: x[1])  # sort by head first
                    head_scores[c][metric][rel].sort(key=lambda x: x[0])  # sort by layer

    return head_scores

def get_checkpoint2dll(model_name, dll_dir, corpora, ablation_mode,
                       ) -> Dict[int, Dict[str, Tuple[int, int, float, float]]]:
    step_id = get_step_id(model_name)
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    checkpoint2dll = dict()
    dll_fps = glob.glob(os.path.join(dll_dir, f"model_deltas_{model_name}-{step_name}*@*{ablation_mode}.csv"))
    for fp in dll_fps:
        fn = fp.split(os.path.sep)[-1]
        l, h, mode = fn.split('.')[0].split('@')[-1].split('-')  # model-name-checkpoint-loss@layer-head-mode.tsv
        if ablation_mode:
            assert mode == ablation_mode
        df = pd.read_csv(os.path.join(fp), index_col=0)
        for corpus in corpora:
            for i in df.loc[df['corpus'] == corpus].index:
                i -= 1
                c = int(re.sub(r'[a-z]', '', df.iloc[i,]['model'].split('-')[step_id]))
                dll_mean, dll_se = float(df.iloc[i,]['delta_test_mean']), float(df.iloc[i,]['delta_test_sem'])
                if c not in checkpoint2dll:
                    checkpoint2dll[c] = dict()
                checkpoint2dll[c][corpus] = checkpoint2dll[c].get(corpus, []) + \
                                            [(int(l), int(h), dll_mean, dll_se)]
        for c in checkpoint2dll.keys():
            for corpus in corpora:
                checkpoint2dll[c][corpus].sort(key=lambda x: x[1])
                checkpoint2dll[c][corpus].sort(key=lambda x: x[0])
    return checkpoint2dll

def get_checkpoint2loss(model_name, loss_dir, ablation_mode, ctx_size=1024, stride=512,
                        ) -> Dict[int, Tuple[int, int, float, float]]:
    """

    :param model_name:
    :param loss_dir:
    :param ablation_mode:
    :param step:
    :param step_id:
    :return: {checkpoint: [
                (checkpoint, layer, head, perplexity, ICL),
                (...),
                ...
             ],
             checkpoint: ...
             }
    """
    checkpoint2loss = dict()
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    step_id = get_step_id(model_name)
    loss_fps = glob.glob(os.path.join(loss_dir, f"{model_name}-{step_name}*-{ctx_size}-{stride}-loss@*-{ablation_mode}.tsv"))
    for fp in loss_fps:
        with open(fp) as f:
            ppl, icl, ppl_std, icl_std = f.readlines()[0].strip().split('\t')
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        l, h, mode = fn.split('.')[0].split('@')[-1].split('-')  # model-name-checkpoint-loss@layer-head-mode.tsv
        if ablation_mode:
            assert mode == ablation_mode
        checkpoint2loss[c] = checkpoint2loss.get(c, []) + [(int(l), int(h), float(ppl), float(icl))]
    for key in checkpoint2loss.keys():
        checkpoint2loss[key].sort(key=lambda x: x[1])
        checkpoint2loss[key].sort(key=lambda x: x[0])
    return checkpoint2loss

def get_checkpoint2head_score(model_name, head_type='induction',
                              ) -> Dict[float, Tuple[int, int, float]]:
    """
    :param model_name:
    :return: {checkpoint: (layer, head, score), checkpoint: (layer, head, score), ...}
    """
    HEAD_DIR = os.path.join(DATA_DIR, f"{head_type}_heads")
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    step_id = get_step_id(model_name)
    head_fps = glob.glob(os.path.join(HEAD_DIR, f"{model_name}-{step_name}*-{head_type}_heads.tsv"))

    checkpoint2score = dict()
    for fp in head_fps:
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        with open(os.path.join(HEAD_DIR, fn)) as f:
            heads = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        for head_idx, score in heads:
            l, h = [int(item) for item in head_idx.split('-')]
            checkpoint2score[c] = checkpoint2score.get(c, []) + [(l, h, float(score))]
        checkpoint2score[c].sort(key=lambda x:x[1])
        checkpoint2score[c].sort(key=lambda x:x[0])
    return checkpoint2score


def get_checkpoint2head_score_others(model_name, head_type='other',
                              ) -> Dict[float, Tuple[int, int, float]]:
    """
    :param model_name:
    :return: {checkpoint: (layer, head, score), checkpoint: (layer, head, score), ...}
    """
    HEAD_DIR = os.path.join(DATA_DIR, f"{head_type}_heads")
    step_name = 'step' if 'pythia' in model_name else 'checkpoint-'
    step_id = get_step_id(model_name)
    head_fps = glob.glob(os.path.join(HEAD_DIR, f"{model_name}-{step_name}*-{head_type}_heads.tsv"))

    checkpoint2score = dict()
    for fp in head_fps:
        fn = fp.split(os.path.sep)[-1]
        c = int(re.sub(r'[a-z]', '', fn.split('-')[step_id]))
        with open(os.path.join(HEAD_DIR, fn)) as f:
            heads = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        for head_idx, s_ratio, log_freq, absp_pos, absp_freq, relp_pos, relp_freq, total, conf in heads:
            l, h = [int(item) for item in head_idx.split('-')]
            checkpoint2score[c] = checkpoint2score.get(c, []) + [(l, h, float(s_ratio), float(log_freq))]
        checkpoint2score[c].sort(key=lambda x:x[1])
        checkpoint2score[c].sort(key=lambda x:x[0])
    return checkpoint2score

def get_sudden_rise(icl_scores, threshold=0.2):
    deltas = [0] + [icl_scores[i]-icl_scores[i-1] > threshold for i in range(1, len(icl_scores))]
    if True in deltas:
        return deltas.index(True)  # first bump larger than threshold, otherwise -1
    return 0

def get_break_through(scores, k=1, mode='max', threshold=None):
    # if first:
    #     maxes = [0] + [max(scores[:i]) for i in range(1, len(scores))]
    #     deltas = [scores[i] - maxes[i] for i in range(len(scores))]
    if threshold:
        return [score > threshold for score in scores].index(True)
    else:
        deltas = [0]*k + [(max(0, scores[i+k]-scores[i]))-(max(0, scores[i]-scores[i-k])) for i in range(k, len(scores)-k)] + [0]*k
    assert len(deltas) == len(scores)
    if mode=='max':
        return np.argmax(deltas)
    elif mode=='min':
        return np.argmin(deltas)

def get_colors_from_cmap(cmap_name, k):
    cmap = plt.get_cmap(cmap_name)  # Get the colormap
    return [cmap(i / (k - 1)) for i in range(k)]  # Sample k colors equally spaced

def plot_unablated(
        token_counts: List[str], scores: List[Dict[str, List[int]]], metrics,
        labels: List[str], colors, markers, markers2=None, highlight=False, mark=True,
        families='', ylim=None, log=False, figsize=(6.5, 3), linewidth=2, markersize=8,
        xname=None, yname=None, markeredgewidth=1, titles=None, prune=False,
        break_legend=False, return_handles=False, diff=False, marker_shapes=('*', '^')
):
    all_handles = []
    num_metrics = len(metrics)
    num_families = max(1, len(families))
    if num_metrics < num_families:
        num_row, num_col = num_metrics, num_families
        family_per_row = False
    else:
        num_row, num_col = num_families, num_metrics
        family_per_row = True
    fig, axs = plt.subplots(
        num_row, num_col,
        # figsize=(4 * num_col, 4 * num_row)
        figsize=figsize
    )
    if 1 in [num_row, num_col]:
        axs = [axs]
    if num_row == 1 and num_col == 1:
        axs = [axs]
    num_subplots = num_row*num_col
    if markers and type(markers[0]) != list and num_subplots > 1:
        markers = [markers]*num_subplots
        if markers2:
            markers2 = [markers2]*num_subplots
    elif num_subplots == 1:
        markers = [markers]
        if markers2:
            markers2 = [markers2]
    for row in axs:
        for ax in row:
            ax.grid(True)
            if ylim:
                ax.set_ylim(ylim)
            if log:
                ax.set_yscale('log')
    cell_id = -1  # to start from 0
    for family_id, family in enumerate(families):
        model_ids = [labels.index(l) for l in labels if family in l]
        for model_id in model_ids:
            model = scores[model_id]
            steps = sorted(list(model.keys()))
            for metric_id, metric in enumerate(metrics):
                cell_id += 1
                cell_id %= num_subplots
                linestyle = 'solid'
                row_id, col_id = (family_id, metric_id) if family_per_row else (metric_id, family_id)
                line = [model[step][metric] for step in steps]
                if type(line[0]) not in [int, float]: line = [item[0] for item in line]
                if len(model) == 30:
                    print(markers, cell_id, model_id)
                    bump_id = markers[cell_id][model_id]
                    if mark and bump_id:
                        axs[row_id][col_id].plot(
                            bump_id, line[bump_id], marker=marker_shapes[0], markersize=markersize,
                            markeredgewidth=markeredgewidth,
                            markerfacecolor=colors[model_id], markeredgecolor='white',
                        )
                        if markers2:
                            bump_id2 = markers2[cell_id][model_id]
                            if bump_id2:
                                axs[row_id][col_id].plot(
                                    bump_id2, line[bump_id2], marker=marker_shapes[1], markersize=int(markersize*0.6),
                                    markeredgewidth=markeredgewidth,
                                    markerfacecolor=colors[model_id], markeredgecolor='white',
                                )
                    if diff and model_id != 0:
                        base = [scores[model_ids[0]][step][metric][0] for step in sorted(list(scores[model_ids[0]].keys()))]
                        axs[row_id][col_id].plot(
                            token_counts, [l-b for l, b in zip(line, base)],
                            color=colors[model_id],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=labels[model_id])
                    else:
                        axs[row_id][col_id].plot(
                            token_counts, line,
                            color=colors[model_id],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=labels[model_id])
                else:
                    bump_id = indices[markers[cell_id][model_id]]
                    if mark and markers[cell_id][model_id] is not None:
                        axs[row_id][col_id].plot(
                            bump_id, line[markers[cell_id][model_id]],
                            marker=marker_shapes[0], color=colors[model_id], markersize=markersize,
                            markeredgewidth=markeredgewidth,
                            markerfacecolor=colors[model_id], markeredgecolor='white',
                        )
                        if markers2:
                            bump_id2 = indices[markers2[cell_id][model_id]]
                            if bump_id2:
                                axs[row_id][col_id].plot(
                                    bump_id2, line[markers2[cell_id][model_id]], marker=marker_shapes[1], markersize=int(markersize*0.6),
                                    markeredgewidth=markeredgewidth,
                                    markerfacecolor=colors[model_id], markeredgecolor='white',
                                )
                    if diff and model_id != 0:
                        base = [scores[model_ids[0]][step][metric][0] for step in sorted(list(scores[model_ids[0]].keys()))]
                        axs[row_id][col_id].plot(
                            token_counts, [l-b for l, b in zip(line, base)],
                            color=colors[model_id],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=labels[model_id]
                        )
                    else:
                        axs[row_id][col_id].plot(
                            indices, line,
                            color=colors[model_id],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=labels[model_id]
                        )
                title = metric if family_per_row else proper_model_name[family]
                if title in corpora_labels:
                    title = corpora_labels[title]
                if titles:
                    title = titles[cell_id]
                if row_id == 0:
                    axs[row_id][col_id].set_title(title)

                axs[row_id][col_id].set_xticks(range(len(token_counts)), rotation=90, size=4,
                                               rotation_mode='anchor', ha='right', va='center')
                if prune:
                    axs[row_id][col_id].set_xticklabels(
                        [item if t%2==1 else '' for t, item in enumerate(token_counts)], rotation=90, size=6,
                        rotation_mode='anchor', ha='right', va='center')
                else:
                    axs[row_id][col_id].set_xticklabels(
                        token_counts, rotation=90, size=4,
                        rotation_mode='anchor', ha='right', va='center')

                axs[row_id][col_id].tick_params('x', pad=0.1)
                axs[row_id][col_id].tick_params('y', pad=0.1)
                # for t, label in enumerate(axs[row_id, col_id].get_xticklabels()):
                #     label.set_text(token_counts[t])
                #     label.set_rotation(90)  # Set rotation angle
                #     label.set_rotation_mode('anchor')  # Set rotation mode
                #     label.set_ha('right')  # Set horizontal alignment to right
                #     label.set_va('center')  # Set vertical alignment to center
                # axs[row_id][col_id].tick_params('y', pad=0)
                for label in axs[row_id][col_id].get_yticklabels():
                    label.set_fontsize(6)
                    label.set_rotation(60)  # Set rotation angle
                    label.set_rotation_mode('default')  # Set rotation mode
                    label.set_ha('right')  # Set horizontal alignment to right
                    label.set_va('center')  # Set vertical alignment to center

                if highlight and markers[cell_id][model_id] is not None:
                    axs[row_id][col_id].axvline(x=bump_id, color=colors[model_id], linestyle='-',
                                           linewidth=10, alpha=0.2, label=None)

            if break_legend and not family_per_row:
                axs[-1][col_id].legend()
            elif not break_legend and not family_per_row:
                h, l = axs[-1][col_id].get_legend_handles_labels()
                all_handles.extend(h)
        if break_legend and family_per_row:
            axs[row_id][-1].legend()
        elif not break_legend and family_per_row:
            h, l = axs[row_id][-1].get_legend_handles_labels()
            print(f'{row_id}th addition: {len(all_handles)}')
            all_handles.extend(h)
    if not break_legend:
        print(f'showing legend: {len(all_handles)}')
        axs[-1][-1].legend(all_handles, labels)
    if yname:
        fig.suptitle(f'{yname} vs Training Steps')
        fig.supylabel(yname)
    if xname:
        fig.supxlabel(xname,y=0.17)
    if not yname and not xname:
        fig.suptitle(f'{metric} vs training steps')
        fig.supylabel(f'{metric}')
    for row in axs:
        for ax in row:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.002)

    plt.show()
    if return_handles:
        return all_handles


token_counts = ["0.5M", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M"]\
    + [f'{i / 10}B' for i in range(5, 101, 5)]
token_counts_int = [500_000*(2**i) for i in range(10)]\
    + [500_000_000*i for i in range(1, 21)]
indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 21, 25, 29]

"""All MODELS"""
models = [
    # 'gpt2-mlp-0-layers',
    # 'gpt2-mlp-1-layers',
    'gpt2-mlp-2-layers',
    'gpt2-mlp-l2-b4-cir3',
    'gpt2-mlp-l2-b4-cir2',
    # 'gpt2-mlp-l2-b4-cir2-s1',
    'gpt2-mlp-l2-b4-r3',
    'gpt2-mlp-l2-b4-r2',
    # 'gpt2-mlp-l2-b4-r2-s1',
    # 'gpt2-mlp-l2-b16-r0',
    # 'gpt2-mlp-l2-b64-r0',
    # 'EleutherAI-pythia-70m',
    # 'EleutherAI-pythia-160m',
    # 'EleutherAI-pythia-410m',
]
ablation_mode = 'pp'
# model_names, ablations, baselines, unique_models, dlls, losses, pmss, ptss, others = [], [], [], [], [], [], [], [], []
model_names, ablations, baselines, = [], [], []
# for each model, we need (1) prefix matching score, (2) previous_token score,
# (3) ICL score and loss with each head ablated, and (4) DLL with each head ablated

for i in tqdm.tqdm(range(len(models))):
    model = models[i]
    step_id = get_step_id(model)
    step = 15625 if 'gpt' in model else 1000
    # # (1) prefix matching score
    # pms = get_checkpoint2head_score(model, head_type='induction')
    # pmss.append(pms)
    # # (2) previous token score
    # pts = get_checkpoint2head_score(model, head_type='previous_token')
    # ptss.append(pts)
    # # (2.5) other heads
    # other = get_checkpoint2head_score_others(model)
    # others.append(other)
    # # (3) ICL score and loss
    # loss_icl = get_checkpoint2loss(model, ablation_mode=ablation_mode, loss_dir=LOSS_DIR)
    # losses.append(loss_icl)
    # # (4) DLL
    # dll = get_checkpoint2dll(model, dll_dir=DLL_DIR, corpora=corpora, ablation_mode=ablation_mode)
    # dlls.append(dll)
    # (5) ablated
    ablation = get_head_scores(model, data_dir=DATA_DIR, corpora=corpora)
    ablations.append(ablation)
    # (6) unablated
    baseline = get_unablated(model, data_dir=DATA_DIR, corpora=corpora)
    # add best PMS to baseline
    try:
        baseline = get_best_pms_per_step(baseline, ablation)
    except:
        pass
    baselines.append(baseline)

def scatter_plot(x, y, alpha=0.7, color=None, cmap=None, figsize=(2,3.5), size=1,
                 xlabel=None, ylabel=None, title=None, legend='Prefix-Matching Score',
                 xlim=None, ylim=None, corpora=None, show_zero_h=True, show_zero_v=True,
                 show_ax_title=True):
    if corpora and len(corpora)>1:
        fig, axs = plt.subplots(nrows=1, ncols=len(corpora) if corpora else 1,
                                # figsize=(4*len(corpora), 4),
                                figsize=figsize,
                                constrained_layout=True)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1,
                                # figsize=(4, 4),
                                figsize=figsize,
                                )
        axs = [axs]
        x = [x]
        y = [y]
    for i in range(len(axs)):
        if not cmap:
            sc = axs[i].scatter(x=x[i], y=y[i], alpha=alpha, s=size)
        else:
            sc = axs[i].scatter(x=x[i], y=y[i], alpha=alpha, s=size, c=color, cmap=cmap)
        if corpora and show_ax_title:
            axs[i].set_title(corpora[i])
        axs[i].grid(True, alpha=0.4)
        if show_zero_h:
            axs[i].axhline(y=0, color='grey', alpha=0.7, linestyle='dashed')
        if show_zero_v:
            axs[i].axvline(x=0, color='grey', alpha=0.7, linestyle='dashed')
    if cmap:
        cbar = fig.colorbar(sc, ax=axs, orientation='vertical', shrink=1, aspect=20)
        cbar.set_label(legend)
    # colorbar = plt.colorbar(sc)
    # colorbar.set_label(legend)  # Label for color scale
    # fig.suptitle(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    fig.supxlabel(xlabel, y=0.06)
    fig.supylabel(ylabel, x=0.06)

def phase_transition_hypothesis(baselines, bumps, corpus):
    peak_dlls = []
    head_bumps = []
    for bump, baseline in zip(bumps, baselines):
        peak_id = np.argmax([baseline[step][corpus][0] for step in sorted(list(baseline.keys()))])
        if len(baseline) == 15:  # pythia
            peak_dlls.append(token_counts_int[indices[peak_id]])
            head_bumps.append(token_counts_int[indices[bump]])
        else:  # gpt
            peak_dlls.append(token_counts_int[peak_id])
            head_bumps.append(token_counts_int[bump])
    return head_bumps, peak_dlls

model_idx=0
step, tok = 15625, '64M'
step, tok = 31250, '128M'
step, tok = 62500, '256M'
step, tok = 2441400, '10B'
step, tok = 2441406, '10B'
step, tok = 128, '0.25B'
step, tok = 256, '0.5B'
step, tok = 512, '1B'
step, tok = 1000, '2B'
step, tok = 2000, '4B'
step, tok = 3000, '6B'
step, tok =4000, '8B'
step,tok =5000, '10B'

# icl_bump_ids = [
#     get_sudden_rise(
#         [
#             model[step]['icl'] for step in sorted(list(model.keys()))
#         ],
#         threshold=0.15)
#     for model in baselines
# ]

icl_bump_ids = [
    get_break_through(
        [
            model[step]['best_pms'] for step in sorted(list(model.keys()))
        ],
        k=1,
        mode='max',
        threshold=0.1
    )
    for model in baselines[1:]
]

uas_bump_ids = [
    get_break_through(
        [
            model[step]['uas_unscaled'] for step in sorted(list(model.keys()))
        ],
        k=1,
        mode='max',
        threshold=0.1
    )
    for model in baselines
]

# icl_bump_ids = [None] + icl_bump_ids
# baselines[-1][2000]

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": ["Computer Modern"], # Set to Computer Modern
    "text.usetex": True,
    "font.size": 12,
    #    'hatch.linewidth': 0.1
})

plot_unablated(token_counts=token_counts, scores=baselines, metrics=['loss'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               # families=['gpt', 'pythia'],
               families=['gpt'],
               prune=True,
               markers=icl_bump_ids,
               log=True, mark=False, figsize=(3,2))

plot_unablated(token_counts=token_counts, scores=baselines, metrics=['best_pms'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               # families=['gpt', 'pythia'],
               families=['t'],
               prune=True,
               markers=icl_bump_ids, log=False, mark=True, markersize=12)

plot_unablated(token_counts=token_counts, scores=baselines, metrics=['icl'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               families=['gpt', 'pythia'],
               # families=['t'],
               prune=True, markers2=None,
               markers=icl_bump_ids, log=False, mark=True, markersize=12)

plot_unablated(token_counts=token_counts, scores=baselines, metrics=['blimp'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               families=['gpt', 'pythia'], markers=icl_bump_ids, log=False, mark=False)

# ICL vs PMS, appendix
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['icl'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt', 'pythia'],
    prune=True, markers2=None,
    markers=icl_bump_ids, log=False, mark=True, markersize=14, return_handles=True,
    figsize=(3.5, 2.5), linewidth=1.2,
    markeredgewidth=1, yname='ICL', xname="Pretraining Tokens",
    break_legend=False,diff=False
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'GPT2 (1 layer)',
        'GPT2 (2 layers)',
        'Pythia-70M',
        'Pythia-160M',
        'Pythia-410M'
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(-0.26, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.8,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 5)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])
plt.savefig("PMS_ICL.pgf", format='pgf', bbox_inches='tight')


# UAS vs BLiMP, appendix

handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['blimp'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt', 'pythia'],
    prune=True, markers2=None,
    markers=uas_bump_ids, log=False, mark=True, markersize=9, return_handles=True,
    figsize=(3.5, 2.5), linewidth=1.2, marker_shapes=['^'],
    markeredgewidth=1, yname='BLiMP', xname="Pretraining Tokens",
    break_legend=False,diff=False
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'GPT2 (1 layer)',
        'GPT2 (2 layers)',
        'Pythia-70M',
        'Pythia-160M',
        'Pythia-410M'
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(-0.26, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.8,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 5)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])
plt.savefig("UAS_BLiMP.pgf", format='pgf', bbox_inches='tight')


# SUPPRESSED: ICL, appendix
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['icl'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt'],
    prune=True, markers2=None,
    markers=icl_bump_ids, log=False, mark=False, markersize=14, return_handles=True,
    figsize=(3, 2.5), linewidth=1.2,
    markeredgewidth=1, yname='ICL', xname="Pretraining Tokens",
    break_legend=False,diff=False, titles=['']
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'NoReg',
        'CopyReg ($\lambda$=0.001)',
        'CopyReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(0.4, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.2,
    handlelength=1,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 3)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])

plt.savefig("ICL_reg.pgf", format='pgf', bbox_inches='tight')


# SUPPRESSED: PMS, appendix
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['best_pms'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt'],
    prune=True, markers2=None,
    markers=icl_bump_ids, log=False, mark=False, markersize=14, return_handles=True,
    figsize=(3, 2.5), linewidth=1.2,
    markeredgewidth=1, yname='Best PMS', xname="Pretraining Tokens",
    break_legend=False,diff=False, titles=['']
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'NoReg',
        'CopyReg ($\lambda$=0.001)',
        'CopyReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(0.4, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.2,
    handlelength=1,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 3)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])

plt.savefig("PMS_reg.pgf", format='pgf', bbox_inches='tight')


# SUPPRESSED: BLIMP, appendix
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['blimp'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt'],
    prune=True, markers2=None,
    markers=icl_bump_ids, log=False, mark=False, markersize=14, return_handles=True,
    figsize=(3, 2.5), linewidth=1.2,
    markeredgewidth=1, yname='BLiMP', xname="Pretraining Tokens",
    break_legend=False,diff=False, titles=['']
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'NoReg',
        'SyntaxReg ($\lambda$=0.001)',
        'SyntaxReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(0.4, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    handlelength=1,
    # labelspacing=-0.2,
    columnspacing=0.2,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 3)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])

plt.savefig("BLiMP_reg.pgf", format='pgf', bbox_inches='tight')



# SUPPRESSED: UAS, appendix
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['uas_unscaled'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt'],
    prune=True, markers2=None,
    markers=icl_bump_ids, log=False, mark=False, markersize=14, return_handles=True,
    figsize=(3, 2.5), linewidth=1.2,
    markeredgewidth=1, yname='UAS', xname="Pretraining Tokens",
    break_legend=False,diff=False, titles=['']
)
plt.tight_layout()
plt.suptitle('')
plt.subplots_adjust(top=0.9, bottom=0.38, wspace=0.2)  # Adjust these values - smaller = less space

l = plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'NoReg',
        'SyntaxReg ($\lambda$=0.001)',
        'SyntaxReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(0.4, -0.4),  # Move legend below x-tick labels
    ncol=3,
    handletextpad=0.3,
    labelspacing=-0.2,
    columnspacing=0.2,
    handlelength=1,
    fontsize='8')  # Position relative to plot
colors = get_colors_from_cmap('viridis', 3)
for i, h in enumerate(l.legendHandles):
    h.set_color(colors[i])

plt.savefig("UAS_reg.pgf", format='pgf', bbox_inches='tight')


# initial 6 models' DLL and phase transition
# icl_bump_ids[0] = icl_bump_ids[1] = None  # setting 0 and 1 layers off
icl_bump_ids = [0,0,0,0,0]
icl_bump_ids = [None]*len(models)
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['dundee', 'meco', 'ns', 'provo'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    # families=['gpt', 'pythia'],
    families=['t'],
    markers=icl_bump_ids, log=False, mark=False,
    figsize=(7,2), linewidth=1.2, markersize=11,
    markeredgewidth=1, yname='$\Delta$LL',
    break_legend=False,
    prune=True, return_handles=True, diff=False
)
plt.tight_layout()
plt.suptitle('')
# handles, _ = plt.gca().get_legend_handles_labels()
plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.2)  # Adjust these values - smaller = less space
# plt.legend(
#     handles = handles,
#     # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
#     labels = [
#         'NoReg',
#         'CopyReg ($\lambda$=0.01) Seed 42',
#         'CopyReg ($\lambda$=0.01) Seed 1',
#         'SyntaxReg ($\lambda$=0.01) Seed 42',
#         'SyntaxReg ($\lambda$=0.01) Seed 1',
#     ],
#     loc="upper center",  # Place legend above plot area
#     bbox_to_anchor=(-1.5, -0.3),  # Move legend below x-tick labels
#     ncol=5,
#     handletextpad=0.3,
#     # labelspacing=-0.2,
#     columnspacing=0.8,
#     fontsize='8')  # Position relative to plot
plt.legend(
    handles = handles,
    # labels = ['GPT2 (0 layer)', 'GPT2 (1 layer)', 'GPT2 (2 layers)', 'Pythia-70M', 'Pythia-160M', 'Pythia-410M'],
    labels = [
        'NoReg',
        'CopyReg ($\lambda$=0.001)',
        'CopyReg ($\lambda$=0.01)',
        'SyntaxReg ($\lambda$=0.001)',
        'SyntaxReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(-1.5, -0.3),  # Move legend below x-tick labels
    ncol=5,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.8,
    fontsize='8')  # Position relative to plot

plt.savefig("reg_trajectory.pgf", format='pgf', bbox_inches='tight')

### small bar graph for final state ###
w = 0.8
e = 0.001
def reg_effect(baselines, corpora, models, figsize=(3.5, 2),
               diff=False, vertical=True, error=2):
    if vertical:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize)
    model_ids = range(1,len(baselines)) if diff else range(len(baselines))
    for i, ax in enumerate(axs):
        ax.grid(True)
        ax.set_axisbelow(True)
        corpus = corpora[i]
        corpus_label = corpora_labels[corpus]
        bars = [baselines[i][sorted(baselines[i].keys())[-1]][corpus][0] for i in model_ids]
        errors = [baselines[i][sorted(baselines[i].keys())[-1]][corpus][1]*error for i in model_ids]
        if diff:
            noreg = [baselines[0][sorted(baselines[0].keys())[-1]][corpus][0] for i in model_ids]
            bars = [reg - noreg for reg, noreg in zip(bars, noreg)]
        if vertical:
            ax.bar(
                height=bars,
                x=model_ids,
                width=w,
                yerr=errors,
                color=get_colors_from_cmap('viridis', len(list(model_ids))),
                label=[models[i] for i in model_ids],
            )
        else:
            ax.barh(
                width=bars,
                y=model_ids,
                height=w,
                xerr=errors,
                color=get_colors_from_cmap('viridis', len(list(model_ids))),
                label=[models[i] for i in model_ids],
            )
        # ax.set_xticklabels(['']*len(baselines))
        if diff:
            allreg = [baselines[i][sorted(baselines[i].keys())[-1]][corpus][0] for i in range(len(baselines)) for corpus in corpora]
            allnoreg = [baselines[0][sorted(baselines[0].keys())[-1]][corpus][0] for i in range(len(baselines)) for corpus in corpora]
            alldiff = [reg - noreg for reg, noreg in zip(allreg, allnoreg)]
            if vertical:
                ax.set_ylim((min(alldiff)-e, max(alldiff)+e))
            else:
                ax.set_xlim((min(alldiff)-e, max(alldiff)+e))
        else:
            if vertical:
                ax.set_ylim((min(bars)-e, max(bars)+e))
            else:
                ax.set_xlim((min(bars)-e, max(bars)+e))
        if not diff:
            if vertical:
                ax.axhline(bars[0], color='red', linestyle='dashed')
            else:
                ax.axvline(bars[0], color='red', linestyle='dashed')
        if vertical:
            ax.set_xlabel(corpus_label)
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        else:
            ax.set_ylabel(corpus_label)
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            for t, tick in enumerate(ax.xaxis.get_major_ticks()):
                if i == 0 and t < 2:  # turn off the first tick to avoid interference with y label
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=6)
    if vertical:
        fig.supylabel('$\Delta$LL')
    else:
        fig.supxlabel('$\Delta$LL', y=0.18)


reg_effect(baselines, corpora, models, diff=False, vertical=False,figsize=(3.5,3), error=0.5)
plt.subplots_adjust(top=0.95, bottom=0.3, wspace=0.2, hspace=0.6)  # Adjust these values - smaller = less space
# # plt.xticks(
# #     ticks = range(len(baselines)),
# #     labels=['']*len(baselines))
# plt.legend(
#     labels=[
#         'NoReg',
#         'CopyReg ($\lambda$=0.01) Seed 42',
#         'CopyReg ($\lambda$=0.01) Seed 1',
#         'SyntaxReg ($\lambda$=0.01) Seed 42',
#         'SyntaxReg ($\lambda$=0.01) Seed 1',
#     ],
#     loc="upper center",  # Place legend above plot area
#     # bbox_to_anchor=(-0.25, -0.2),  # Move legend below x-tick labels
#     bbox_to_anchor=(0.48, -0.6),  # Move legend below x-tick labels
#     ncol=2,
#     handletextpad=0.3,
#     # labelspacing=-0.2,
#     columnspacing=0.8,
#     fontsize='8')  # Position relative to plot

plt.legend(
    labels=[
        'NoReg',
        'CopyReg ($\lambda$=0.001)',
        'CopyReg ($\lambda$=0.01)',
        'SyntaxReg ($\lambda$=0.001)',
        'SyntaxReg ($\lambda$=0.01)',
    ],
    loc="upper center",  # Place legend above plot area
    # bbox_to_anchor=(-0.25, -0.2),  # Move legend below x-tick labels
    bbox_to_anchor=(0.48, -1.2),  # Move legend below x-tick labels
    ncol=2,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.8,
    fontsize='8')  # Position relative to plot

# plt.savefig("reg_effect_10b.pgf", format='pgf', bbox_inches='tight')

baselines[0][max(list(baselines[0].keys()))]
# initial 6 models' DLL and phase transition
# icl_bump_ids[0] = icl_bump_ids[1] = None  # setting 0 and 1 layers off
# icl_bump_ids.insert(0, None)
# uas_bump_ids.insert(0, None)
handles = plot_unablated(
    token_counts=token_counts, scores=baselines, metrics=['dundee', 'meco', 'ns', 'provo'],
    labels=models, colors=get_colors_from_cmap('viridis', len(models)),
    families=['gpt', 'pythia'],
    # families=['t'],
    # markers=[0,0,0],
    markers=icl_bump_ids,
    markers2=uas_bump_ids,
    log=False, mark=True,
    figsize=(7,4), linewidth=1.2, markersize=14,
    markeredgewidth=1, yname='$\Delta$LL', xname='Pretraining Tokens',
    break_legend=False,
    prune=True, return_handles=True
)
plt.tight_layout()
plt.suptitle('')
# handles, _ = plt.gca().get_legend_handles_labels()
plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.2)  # Adjust these values - smaller = less space
plt.legend(
    handles = handles,
    labels = [
        'GPT2 (0 layer)',
        'GPT2 (1 layer)',
        'GPT2 (2 layers)',
        'Pythia-70M',
        'Pythia-160M',
        'Pythia-410M'
    ],
    loc="upper center",  # Place legend above plot area
    bbox_to_anchor=(-1.5, -0.6),  # Move legend below x-tick labels
    ncol=6,
    handletextpad=0.3,
    # labelspacing=-0.2,
    columnspacing=0.8,
    fontsize='8')  # Position relative to plot

plt.savefig("coinciding.pgf", format='pgf', bbox_inches='tight')

# plt.savefig("sas_suppression.pgf", format='pgf', bbox_inches='tight')


# icl_bump_ids[0] = None
# uas_bump_ids[0] = None
plot_unablated(token_counts=token_counts, scores=baselines, metrics=['uas_unscaled', 'icl'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               # families=['gpt', 'pythia'],
               families=['t'],
               markers=[icl_bump_ids, icl_bump_ids], log=False, mark=True,
               figsize=(3.5,2.5), linewidth=1.2, markersize=11,
               markeredgewidth=1, yname='Score',
               titles=['UAS', 'ICL Score'],
               prune=True)
plt.tight_layout()
plt.suptitle('')
handles, _ = plt.gca().get_legend_handles_labels()
plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.2)  # Adjust these values - smaller = less space
plt.legend(handles=handles,
           labels=['GPT2 (1 layer)',
                   'GPT2 (2 layers)',
                   'Pythia-70M',
                   'Pythia-160M',
                   'Pythia-410M'],
           loc="upper center",  # Place legend above plot area
           bbox_to_anchor=(-0.25, -0.2),  # Move legend below x-tick labels
           ncol=3,
           handletextpad=0.3,
           # labelspacing=-0.2,
           columnspacing=0.8,
           fontsize='8')  # Position relative to plot

plt.savefig("bump.pgf", format='pgf', bbox_inches='tight')

### figure 1 ###

def visualize_phase_transition_hypothesis(
        baselines, corpora, models,
        icl_bump_ids, uas_bump_ids=None,
        log=True, trendline=True,
        show_two_billion_hypothesis=True, show_r=True, markers=['^','*']
):
    bump_ids = [uas_bump_ids, icl_bump_ids] if uas_bump_ids else [icl_bump_ids]
    fig, axs = plt.subplots(len(bump_ids),
                            len(corpora), figsize=(3.5, 1.5*len(bump_ids)))
    for i, bump_id in enumerate(bump_ids):
        for j, corpus in enumerate(corpora):
            axs[i][j].grid(True)
            axs[i][j].set_xlim(7, 10)
            axs[i][j].set_ylim(7, 10)
            bump_toks, dll_toks = phase_transition_hypothesis(baselines, bump_id, corpus)
            x, y = bump_toks, dll_toks
            if log:
                x, y = np.log10(bump_toks), np.log10(dll_toks)
            if trendline:
                axs[i][j].plot((7, 10), np.poly1d(np.polyfit(x, y, 1))((7, 10)),
                            label='Observed Trendline')
            pearson = pearsonr(x, y)
            stats = str(round(pearson.statistic, 3))
            if pearson.pvalue < 0.05:
                stats += '*'
            if pearson.pvalue < 0.01:
                stats += '*'
            if show_two_billion_hypothesis:
                axs[i][j].axhline(y=np.log10(2*(10**9)), linestyle='--', color='grey',
                               label='2B Hypothesis')
                axs[i][j].plot(
                    (7, 10),
                    np.poly1d(np.polyfit([1, 2], [1, 2], 1))((7, 10)),
                    linestyle='--', color='orange', label='PT Hypothesis'
                )
            for k in range(len(models)):
                marker = markers[i]
                size = 80 if i==0 else 160
                axs[i][j].scatter(x[k], y[k], color=colors[k], marker=marker, s=size,
                               edgecolors='white', linewidths=1, zorder=4,
                               label=models[k])
            if show_r:
                axs[i][j].text(8.1, 7.1, f"r = {stats}", size=6)
            if i == 0:
                axs[i][j].set_title(corpora_labels[corpus])
            # x_tick_labels = axs[i].get_xticks()  # Get x ticks (positions)
            # y_tick_labels = axs[i].get_yticks()  # Get y ticks (positions)
            axs[i][j].set_xticks(range(7, 11))
            axs[i][j].set_yticks(range(7, 11))
            axs[i][j].set_xticklabels([f"$10^{{{str(l)}}}$" for l in range(7,11)],
                                   fontsize=5)
            axs[i][j].set_yticklabels([f"$10^{{{str(l)}}}$" for l in range(7,11)],
                                   fontsize=5)
            if j != 0:
                # axs[i].set_yticks([])
                axs[i][j].set_yticklabels([])
            if i != len(bump_ids)-1:
                axs[i][j].set_xticklabels([])
            if i == 0:  # hide all ticks for top row
                for tick in axs[i][j].xaxis.get_major_ticks() + axs[i][j].xaxis.get_minor_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            if j == len(corpora)//2:
                if i == 0:
                    axs[i][j].set_xlabel("Phase Transition := UAS")
                    axs[i][j].xaxis.set_label_coords(0, -0.1)  # Adjust manually
                elif i == 1:
                    axs[i][j].set_xlabel("Phase Transition := PMS", x=0)
                    axs[i][j].xaxis.set_label_coords(0, -0.28)  # Adjust manually

    # fig.supxlabel('Phase Transition')
    fig.supylabel('$\Delta$LL Peak')
colors = get_colors_from_cmap('viridis', 6)
# markers = ['']

# def get_sse(dll_peak_toks, bump_toks, log=True):
#     twob = [2_000_000_000]*len(dll_peak_toks)
#     if log:
#         dll_peak_toks = np.log10(dll_peak_toks)
#         bump_toks = np.log10(bump_toks)
#         twob = np.log10(twob)
#     pt = bump_toks
#     sse_pt = sum([(y-yhat)**2 for y, yhat in zip(dll_peak_toks, pt)])
#     sse_twob = sum([(y-yhat)**2 for y, yhat in zip(dll_peak_toks, twob)])
#     return sse_twob, sse_pt
#
# from scipy.stats import f
#
# bump_toks, dll_toks = phase_transition_hypothesis(baselines, uas_bump_ids, 'provo')
# rss1, rss2 = get_sse(dll_toks, bump_toks)  # 2b, pt
# df1 = len(dll_toks) - 1  # intercept
# df2 = len(dll_toks) - 2  # intercept, slope
# f_stat = ((rss1 - rss2) / (df1 - df2)) / (rss2 / df2)
# p_value = 1 - f.cdf(f_stat, df1 - df2, df2)

visualize_phase_transition_hypothesis(
    baselines,
    # corpora=['dundee','provo','ns'],
    corpora=corpora,
    models=[
        'GPT2 (4)',
        'GPT2 (16)',
        'GTP2 (64)',
        'Pythia-70M',
        'Pythia-160M',
        'Pythia-410M'
    ],
    icl_bump_ids=icl_bump_ids, uas_bump_ids=uas_bump_ids,
    trendline=True, log=True, show_r=True
)
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3, wspace=0.25,
                    hspace=0.3)  # Adjust these values - smaller = less space

handles, labels = plt.gca().get_legend_handles_labels()
new_handles = []
for i, handle in enumerate(handles):
    if i < 3:
        new_handles.append(handle)
        continue
    # Get the color of original handle
    fc = handle.get_facecolor()
    # Create a Line2D object for the legend
    new_handle = plt.Line2D([], [],
                            marker='s',  # New marker style
                            markersize=9,  # Marker size
                            linestyle='none',  # No line
                            markerfacecolor=fc[0],  # Use same color as scatter
                            markeredgecolor='white',  # Use same color for edge
                            markeredgewidth=2)
    new_handles.append(new_handle)

plt.legend(
    handles=new_handles,
    labels=labels,
    loc="upper center",  # Place legend above plot area
    # bbox_to_anchor=(-0.8, -0.3),  # Move legend below x-tick labels
    bbox_to_anchor=(-1.5, -0.45),  # Move legend below x-tick labels
    ncol=3,
    # handletextpad=0.3,
    labelspacing=0.5,
    columnspacing=0.8,
    markerscale=0.8,
    fontsize='6')  # Position relative to plot
# plt.tight_layout()

plt.savefig('splashy.pgf', format='pgf')



plot_unablated(token_counts=token_counts, scores=baselines, metrics=['uas_unscaled'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               families=['gpt', 'pythia'], icl_bump_ids=icl_bump_ids, log=False, mark=False)

plot_unablated(token_counts=token_counts, scores=baselines, metrics=['icl'],
               labels=models, colors=get_colors_from_cmap('viridis', len(models)),
               families=['gpt', 'pythia'], icl_bump_ids=icl_bump_ids, log=False, mark=False)

# sorted(dlls[model_idx][step]['dundee'], key=lambda x: x[2], reverse=True)[:3]

# create table of Pearson
pearsons = dict()
model_ids = [models.index(model) for model in ['gpt2-mlp-1-layers', 'gpt2-mlp-2-layers',
                                               'EleutherAI-pythia-70m','EleutherAI-pythia-160m',
                                               'EleutherAI-pythia-410m']
             ]
step_sets = [[3906, 7812, 15625, 31250, 62500],
             [256, 512, 1000, 2000, 3000]]
# [item[2] for item in ablations[1][3906]['dundee']]
# t = pearsonr([item[2] for item in ablations[1][3906]['dundee']], [item[2] for item in ablations[1][3906]['pms']])
pearsonr([item[2] for item in ablations[0][62500]['provo']], [item[2] for item in ablations[0][62500]['pms']])

for idx in model_ids:
    model = models[idx]
    if 'gpt' in models[idx]:
        step_set = step_sets[0]
    elif 'pythia' in models[idx]:
        step_set = step_sets[1]
    if model not in pearsons:
        pearsons[model] = dict()
    for step in step_set:
        if step not in pearsons[model]:
            pearsons[model][step] = dict()
        for corpus in corpora:
            y = [item[2] for item in ablations[idx][step][corpus]]
            for head_score in ['sas_unscaled', 'pms']:
                if head_score not in pearsons[model][step]:
                    pearsons[model][step][head_score] = dict()
                x = ablations[idx][step][head_score]
                if head_score == 'sas_unscaled':
                    x = x['total']
                x = [item[2] for item in x]
                result = pearsonr(x, y)
                # if corpus not in pearsons[model][step][head_score]:
                #     pearsons[model][step][head_score][corpus] = dict()
                pearsons[models[idx]][step][head_score][corpus] = (result.statistic, result.pvalue)

num_steps = len(list(pearsons[list(pearsons.keys())[0]])) * 2  # repeat for sas and pms
header = []
model_row = [f'\multicolumn{{{4}}}{{c|}}{{{models[idx]}}}' for idx in model_ids]
model_row.insert(2, ' ')
model_row.insert(0, ' ')
model_row = '&'.join(model_row) + '\\\\'
corpus_row = [c[:2] for c in corpora]*5
corpus_row.insert(8, ' ')
corpus_row.insert(0, ' ')
corpus_row = '&'.join(corpus_row) + '\\\\'
header = [
    '\\toprule',
    model_row,
    '\\midrule',
    corpus_row,
    '\\midrule'
]
num_gpt = 2
rows = []
step_names = [['16M', '32M', '64M', '128M', '256M'],
              ['0.5B', '1B', '2B', '4B', '6B']]
for head_score in ['sas_unscaled', 'pms']:
    for i, steps in enumerate(zip(step_sets[0], step_sets[1])):
        gpt_step, pythia_step = steps
        if len(rows) == 5:
            rows.append('\\midrule')
        row = []
        for j, idx in enumerate(model_ids):
            model = models[idx]
            # if 'gpt' in model:
            #     step_set = step_sets[0]
            # elif 'pythia' in model:
            #     step_set = step_sets[1]
            if 'gpt' in model:
                step = gpt_step
            elif 'pythia' in model:
                step = pythia_step
            if not row:
                row = [step_names[0][i]]
            if row and j == num_gpt:
                row.append(step_names[1][i])
            for corpus in corpora:
                stats, pval = pearsons[model][step][head_score][corpus]
                stats = round(stats, 2)
                if pval <= 0.05:
                    row.append(f'\\textbf{{\\textcolor{{red}}{{{str(stats)}}}}}')
                else:
                    row.append(str(stats))
        rows.append('&'.join(row) + '\\\\')

rows = ['\\setlength{\\tabcolsep}{.45em}', '\\begin{tabular}{c|rrrr|rrrr|c|rrrr|rrrr|rrrr}']+ header + rows + ['\\bottomrule', '\\end{tabular}']
print('\n'.join(rows))

# scatter plot: prefix-matching score x ICL
scatter_plot(x=[item[2] for item in ablations[model_idx][step]['pms']],
             y=[item[2] - baselines[model_idx][step]['icl'] for item in ablations[model_idx][step]['icl']],
             cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
             xlabel='Prefix-Matching Score', ylabel='Delta In-Context Learning Score', title=f'{models[model_idx]} ({tok} tokens)')

# TODO scatter plot: Delta ICL x DLL (potentially color by pms or pts)
scatter_plot(x=[[item[2] - baselines[model_idx][step]['icl'] for item in ablations[model_idx][step]['icl']]]*len(corpora),
             y=[[item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]] for corpus in corpora],
             corpora=corpora, cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
             xlabel='Delta In-Context Learning Score', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')

for corpus in ['dundee', 'ns', 'meco', 'provo']:
    plt.plot(
        [pearsonr(
            [item[2] for item in ablations[model_idx][step]['icl']],
            [item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]]
        ).statistic for step in [256, 512, 1000, 2000, 3000]], label=corpus
    )
    plt.ylabel("Pearson's R")
    plt.title("Correlation between Delta ICL and Delta DLL")
    plt.xticks(list(range(5)),
               ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])
plt.grid(True)
plt.legend()

# scatter plot: PMS x Delta LL for Pythia 70m at 64M
step = 1000
tok = '2B'
corpora = ['dundee', 'meco', 'provo', 'ns']
corpora = ['dundee']
scatter_plot(x=[[item[2] for item in ablations[model_idx][step]['pms']]] * len(corpora),
             y=[[item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]] for
                corpus in corpora],
             figsize=(3,2),
             alpha=0.7,
             size=20,
             corpora=corpora, color=[item[2] for item in ablations[model_idx][step]['pms']],
             xlabel='PMS', ylabel='$\Delta\Delta$LL',
             # title=f'Pythia-70M ({tok} tokens)',
             title='', show_ax_title=False,
             )
plt.tight_layout()
plt.savefig("pearsons.pgf", format='pgf', bbox_inches='tight')

# scatter plot: PMS x Delta LL
for step, tok in zip(
        [256, 512, 1000, 2000, 3000],
        ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B']):

    scatter_plot(x=[[item[2] for item in ablations[model_idx][step]['pms']]]*len(corpora),
                 y=[[item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]] for corpus in corpora],
                 corpora=corpora, cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
                 xlabel='PMS', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')

for corpus in ['dundee', 'ns', 'meco', 'provo']:
    plt.plot(
        [pearsonr(
            [item[2] for item in ablations[model_idx][step]['pms']],
            [item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]]
        ).statistic for step in [256, 512, 1000, 2000, 3000]], label=corpus
    )
    plt.ylabel("Pearson's R")
    plt.title("Correlation between PMS and Delta DLL")
    plt.xticks(list(range(5)),
               ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])
plt.grid(True)
plt.legend()

# scatter plot: SAS x Delta BLiMP
sas = [item[2] for item in ablations[model_idx][step]['sas_scaled']['total']]
delta_blimp = [item[2] - baselines[model_idx][step]['blimp'] for item in ablations[model_idx][step]['blimp']]
scatter_plot(x=[sas],
             y=[delta_blimp],
             cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
             xlabel='SAS score', ylabel='Delta BLiMP', title=f'{models[model_idx]} ({tok} tokens)')
plt.plot(
    [pearsonr(
        [item[2] for item in ablations[model_idx][step]['sas_scaled']['total']],
        [item[2] for item in ablations[model_idx][step]['blimp']],
    ).statistic for step in [256, 512, 1000, 2000, 3000]]
)
plt.grid(True)
plt.ylabel("Pearson's R")
plt.title("R(SAS, Delta BLiMP)")
plt.xticks(list(range(5)),
           ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])

# scatter plot: Delta BLiMP x DLL (potentially color by pms or pts)
for step, tok in zip(
        [256, 512, 1000, 2000, 3000],
        ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B']):
    scatter_plot(x=[[item[2] - baselines[model_idx][step]['blimp'] for item in ablations[model_idx][step]['blimp']]]*len(corpora),
                 y=[[item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]] for corpus in corpora],
                 corpora=corpora, cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
                 xlabel='Delta BLiMP', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')

for corpus in ['dundee', 'ns', 'meco', 'provo']:
    plt.plot(
        [pearsonr(
            [item[2] for item in ablations[model_idx][step]['blimp']],
            [item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]]
        ).statistic for step in [256, 512, 1000, 2000, 3000]], label=corpus
    )
    plt.ylabel("Pearson's R")
    plt.title("Correlation between Delta BLiMP and Delta DLL")
    plt.xticks(list(range(5)),
               ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])
plt.grid(True)
plt.legend()


plt.grid(True)
plt.ylabel("Pearson's R")
plt.title("Correlation between Delta BLiMP and Delta DLL")
plt.xticks(list(range(5)),
           ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])



# scatter plot: SAS x DLL (potentially color by pms or pts)
for step, tok in zip(
        [256, 512, 1000, 2000, 3000],
        ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B']):
    sas = [item[2] for item in ablations[model_idx][step]['sas_scaled']['case']]
    dlls = [[item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]] for corpus in corpora]
    scatter_plot(x=[sas]*len(corpora),
             y=dlls,
             corpora=corpora, cmap='viridis', color=[item[2] for item in ablations[model_idx][step]['pms']],
             xlabel='SAS score', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')
for corpus in ['dundee', 'ns', 'meco', 'provo']:
    plt.plot(
        [pearsonr(
            [item[2] for item in ablations[model_idx][step]['sas_scaled']['total']],
            [item[2] - baselines[model_idx][step][corpus][0] for item in ablations[model_idx][step][corpus]]
        ).statistic for step in [256, 512, 1000, 2000, 3000]], label=corpus
    )
    plt.ylabel("Pearson's R")
    plt.title("Correlation between Delta DLL and SAS")
    plt.xticks(list(range(5)),
               ['0.5B', '1.0B', '2.0B', '4.0B', '6.0B'])
plt.grid(True)
plt.legend()

# scatter plot: syntax score x DLL
scatter_plot(x=[[item[2] for item in others[model_idx][step]]]*len(corpora),
             y=[[item[2] - baselines[model_idx][step][corpus][0] for item in dlls[model_idx][step][corpus]] for corpus in corpora],
             corpora=corpora, cmap='viridis', color=[item[2] for item in others[model_idx][step]],
             xlabel='Syntax Score', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')

# scatter plot: rare word x DLL
scatter_plot(x=[[item[3] for item in others[model_idx][step]]]*len(corpora),
             y=[[item[2] - baselines[model_idx][step][corpus][0] for item in dlls[model_idx][step][corpus]] for corpus in corpora],
             corpora=corpora, cmap='viridis', color=[item[3] for item in others[model_idx][step]],
             xlabel='Rare Word Score', ylabel='Delta DLL', title=f'{models[model_idx]} ({tok} tokens)')
