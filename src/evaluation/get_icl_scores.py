import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
import torch, argparse, pathlib, pickle
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

data_fp = r"C:\Users\aozsa\Codes\sentence-processing-as-icl\data\cc100_10m_tokens.pkl"

# def chunk_data(ctx_size):
#     with open(data_fp, 'rb') as f:
#         data = pickle.load(f)
#     chunks = []
#     for start_idx in range(0, len(data), ctx_size):
#         chunks.append(data[start_idx:start_idx+ctx_size])
#     return chunks

def get_loss_perplexity(encodings, model, ablation_mode, stride, ctx_size,
                        e1, e2, l1, l2):
    model.to(device)
    loss_fct = CrossEntropyLoss(reduce=False)
    seq_len = len(encodings)
    nlls = []
    icl_means = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + ctx_size, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if 'pythia' in model.config._name_or_path:
            input_ids = encodings[begin_loc:end_loc].unsqueeze(0).to(device)
        else:
            input_ids = encodings[begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:-trg_len] = -100

        with torch.no_grad():
            outputs = model.forward_plus(input_ids, ablation_mode=ablation_mode, labels=target_ids)
            neg_log_likelihood = outputs.loss
        with torch.no_grad():
            outputs = model.forward_plus(input_ids, ablation_mode=ablation_mode, labels=input_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Compute per-token loss using CrossEntropyLoss
        # compute loss for each token
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Print the loss per token (use the tokenizer to display the actual tokens)

        nlls.append(neg_log_likelihood)
        if len(loss_per_token.tolist()) > e1:
            e = np.mean(loss_per_token.tolist()[e1:e2])
            l = np.mean(loss_per_token.tolist()[l1:l2])
            icl_means.append(e-l)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl_mom = torch.exp(torch.stack(nlls).mean())
    ppl_sem = np.std([float(nll) for nll in nlls])
    icl_score_mom = np.mean(icl_means)
    icl_score_sem = np.std(icl_means)

    return ppl_mom, icl_score_mom, ppl_sem, icl_score_sem

def induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head):
    if ablation_head:
        l, h = ablation_head.split('-')
        return {int(l): [int(h)]}
    TSV_DIR = os.path.join(DATA_DIR, 'induction_heads')
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    with open(os.path.join(TSV_DIR, model_name+'-induction_heads.tsv')) as f:
        induction_heads = [line.strip().split('\t')[0] for line in f.readlines() if\
                           line.strip() and float(line.split('\t')[1]) >= ablation_threshold]
        dct = dict()
        for l_h in sorted(induction_heads, key=lambda x:int(x.split('-')[0])):
            l, h = l_h.split('-')
            dct[int(l)] = dct.get(int(l), []) + [int(h)]
    print(f"threshold: {str(ablation_threshold)} ||| ablating these heads: {list(dct.items())}")
    return dct


def write_losses(model_dir, output_dir, results, ctx_size,
                 stride, revision, ablation_threshold, ablation_mode, ablation_head):
    """
    :param model_dir:
    :param output_dir:
    :param results: ppl_mean, icl_mean, ppl_sd, icl_sd
    :param ctx_size:
    :param stride:
    :param revision:
    :param ablation_threshold:
    :param ablation_mode:
    :param ablation_head:
    :return:
    """
    print(f"model: {model_dir}, ppl: {float(results[0])} ({results[2]}), icl:{results[1]} ({results[3]})")

    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    if ablation_mode:  # if ablation
        if ablation_threshold:
            attention_info = str(int(100 * ablation_threshold))  # e.g. surp@20-full
        elif ablation_head:
            attention_info = ablation_head  # e.g. surp@1-1-full (1-th layer, 1-th head, full ablation)
        attention_info += f'-{ablation_mode}'
        suffix = f'loss@{attention_info}'
    else:
        suffix = 'loss'

    with open(os.path.join(output_dir, '-'.join([
        model_name, str(ctx_size), str(int(stride)),
        suffix,
    ])) + '.tsv', 'w') as f:
        f.write('\t'.join([str(float(item)) for item in results]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fp', default=os.path.join(DATA_DIR, 'cc100_10m_tokens.pkl'),
                        help=f"path to token file, default={os.path.join(DATA_DIR, 'rt_data', 'all_toks.tsv')}")
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-at', '--ablation_threshold', type=float, default=None,
                        help="induction heads with prefix matching score above this threshold will be ablated,\
                        default=1.0 (no ablation)")
    parser.add_argument('-am', '--ablation_mode', choices=['full', 'pp'], default=None,
                        help="type of ablation to perform: ['full', 'pp'], default=None")
    parser.add_argument('-ah', '--ablation_head', default=None,
                        help="head to ablate e.g. '-ah 0-3',  default=None")
    parser.add_argument('-r', '--revision', required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-c', '--ctx_size', default=1024,
                        help=f'context LMs will use, default=max=1024')
    parser.add_argument('-s', '--stride', default=None,
                        help=f'stride for moving window, default=context/2')
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'losses'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'losses')}")
    parser.add_argument('-p', '--pre_chunked', action='store_true',
                        help=f"if the val set is pre-chunked, default=False")
    parser.add_argument('-t', '--tokenize', action='store_true',
                        help="tokenize the text, default=False")
    parser.add_argument('-ds', '--data_size', type=int, default=100_000,
                        help=f"size of data to compute loss and ICL from, max=10_000_000, default=100_000")
    parser.add_argument('-e1', '--early_one', type=int, default=40,
                        help="lower bound for the early token range, default=40")
    parser.add_argument('-e2', '--early_two', type=int, default=60,
                        help="second element for ICL score, default=60")
    parser.add_argument('-l1', '--later_one', type=int, default=450,
                        help="lower bound for the early token range, default=450")
    parser.add_argument('-l2', '--later_two', type=int, default=550,
                        help="second element for ICL score, default=550")
    args = parser.parse_args()
    data_fp, model_dir, ablation_threshold, ablation_mode, ablation_head,\
    revision, ctx_size, stride, output_dir, data_size, pre_chunked,\
    e1, e2, l1, l2 =\
        args.data_fp, args.model_dir, args.ablation_threshold, args.ablation_mode, args.ablation_head,\
        args.revision, int(args.ctx_size), args.stride, args.output_dir, args.data_size, args.pre_chunked,\
        args.early_one, args.early_two, args.later_one, args.later_two


    stride = int(ctx_size/2) if not stride else int(stride)  # stride defaults to 50% overlap sliding window
    if ablation_head:
        output_dir = os.path.join(DATA_DIR, 'by_head_ablation', 'losses')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(data_fp, 'rb') as f:
        encodings = pickle.load(f)
    if pre_chunked:  # flatten
        encodings = [idx for batch in encodings for idx in batch]
    print(f"using {data_size} tokens ({round((data_size/len(encodings)*100), 3)}% of val set)")
    encodings = encodings[:data_size]

    encodings = torch.tensor(encodings)
    encodings.to(device)

    induction_heads = []  # initialize
    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            if ablation_mode:
                induction_heads = induction_tsv2dct(os.path.join(model_dir, checkpoint), revision,
                                                    ablation_threshold, ablation_head)
            model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint),
                                                            ablation_head_idx=induction_heads)
            results = get_loss_perplexity(encodings, model, ablation_mode,
                                          stride, ctx_size, e1, e2, l1, l2)
            write_losses(os.path.join(model_dir, checkpoint), output_dir, results,
                         ctx_size, stride, revision, ablation_threshold, ablation_mode, ablation_head)

    elif os.path.isdir(model_dir):  # if one custom model
        if ablation_mode:
            induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
        model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir),
                                                        ablation_head_idx=induction_heads)
        results = get_loss_perplexity(encodings, model, ablation_mode,
                                      stride, ctx_size, e1, e2, l1, l2)
        write_losses(model_dir, output_dir, results, ctx_size, stride,
                     revision, ablation_threshold, ablation_mode, ablation_head)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if ablation_mode:
                induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
            if "gpt" in model_dir:
                # tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
                model = AblationGPT2LMHeadModel.from_pretrained(model_dir,
                                                                ablation_head_idx=induction_heads)
            elif "pythia" in model_dir:
                # tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=induction_heads)
            results = get_loss_perplexity(encodings, model, ablation_mode,
                                          stride, ctx_size, e1, e2, l1, l2)
            write_losses(model_dir, output_dir, results, ctx_size, stride,
                         revision, ablation_threshold, ablation_mode, ablation_head)
        elif 'pythia' in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                if ablation_mode:
                    induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=induction_heads)
                results = get_loss_perplexity(encodings, model, ablation_mode,
                                              stride, ctx_size, e1, e2, l1, l2)
                write_losses(model_dir, output_dir, results, ctx_size, stride,
                             revision, ablation_threshold, ablation_mode, ablation_head)

if __name__ == "__main__":
    main()