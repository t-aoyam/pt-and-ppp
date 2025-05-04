"""
Author: Tatsuya
Given a model, either (1) print the prefix matching score of all heads or
                      (2) print the index of the heads that have a higher score than a given threshold t
Prefix matching score:
Given a repeated random sequence of size s, at position s+i (i < s),
compute a given head's average attention paid to i+1 (same token as the current position's next token)

Logit attribution:
Not implemented yet
"""

import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
import pathlib, argparse, torch, einops
from transformers import (AutoModelForCausalLM, GPTNeoXForCausalLM,
                          AutoTokenizer, GPT2TokenizerFast, AutoConfig)
from torch.nn import CrossEntropyLoss
from torch import cuda
from matplotlib import pyplot as plt

def _generate_random_tokens(tokenizer, seq_len=50, rep=2, num_batches=100):
    size = (num_batches, seq_len)
    input_tensor = torch.randint(0, len(dict(tokenizer.get_vocab())) - 1, size)
    # input_tensor.shape
    # random_tokens = input_tensor.to(model.cfg.device)
    repeated_tokens = einops.repeat(input_tensor, f"batch seq_len -> batch ({rep} seq_len)")
    # repeated_tokens.shape
    return repeated_tokens


def get_scores(model, tokenizer, head_type, num_layers, num_heads, seq_len=50, rep=2, num_batches=100):
    if head_type == 'induction':
        num_tokens_back = seq_len - 1
    elif head_type == 'previous_token':
        num_tokens_back = 1
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    loss_fct = CrossEntropyLoss(reduce=False)
    inputs = _generate_random_tokens(tokenizer, seq_len, rep, num_batches)
    inputs.to(device)
    with torch.no_grad():
        output = model(inputs, labels=inputs, output_attentions=True)
    logits = output.logits
    # logits.shape
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs[..., 1:].contiguous()
    # Compute per-token loss using CrossEntropyLoss
    # compute loss for each token
    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    for i in (99, 991, 99):
        plt.plot(loss_per_token[:99])

    # loss_per_token.shape

    # correct_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
    head2score = dict()  # key (layer, head), value (score)
    for l in range(num_layers):
        for h in range(num_heads):
            head_id = (l, h)
            for b in range(num_batches):
                for source_id in range(seq_len, seq_len*rep):
                    target_id = source_id-(num_tokens_back)
                    head2score[head_id] = head2score.get(head_id, 0) +\
                                          float(output.attentions[l][b][h][source_id][target_id])  # l-th layer, h-th head, attention from source to target
    for head_id in head2score:
        head2score[head_id] /= ((seq_len*rep-seq_len)*num_batches)
    return head2score

def write_heads(model_dir, output_dir, head_type, head2score, revision, threshold):
    induction_heads = []
    config = AutoConfig.from_pretrained(model_dir)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    for l in range(num_layers):
        for h in range(num_heads):
            head_id = (l, h)
            if head2score[head_id] > threshold:
                print(f"model: {model_dir} ||| {head_id}: {head2score[head_id]}")
                head_id_w = '-'.join([str(num) for num in head_id])
                induction_heads.append('\t'.join([head_id_w, str(head2score[head_id])]))
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    with open(os.path.join(output_dir, '-'.join([model_name, f'{head_type}_heads'])) + '.tsv', 'w') as f:
        f.write('\n'.join(induction_heads))

def main():
    ROOT_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-o', '--output_dir', default=None,
                        help=f"output directory")
    parser.add_argument('-t', '--head_type', choices=['induction', 'previous_token'], default='induction',
                        help="type of head to detect, default=induction")
    parser.add_argument('-s', '--seq_len', type=int, default=50,
                        help="length of the random token sequence, default=50")
    parser.add_argument('-p', '--rep', type=int, default=2,
                        help="numbers by which the random sequence is repeated, default=2")
    parser.add_argument('-b', '--num_batches', type=int, default=100,
                        help="how many repeated sequences to test the model on, default=100")
    parser.add_argument('-r', '--revision', type=str, required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-th', '--threshold', type=float, default=0.0,
                        help="threshold of prefix matching score,\
                        beyond which a given head is considered a particular type of head, default=0.8")

    args = parser.parse_args()
    model_dir, output_dir, head_type, seq_len, rep, num_batches, revision, threshold = \
        args.model_dir, args.output_dir, args.head_type, args.seq_len,\
        args.rep, args.num_batches, args.revision, args.threshold
    if not output_dir:
        output_dir = os.path.join(DATA_DIR, f'{head_type}_heads')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Get the number of layers and attention heads
    config = AutoConfig.from_pretrained(model_dir)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(model_dir, checkpoint))
            model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, checkpoint))
            scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                num_layers=num_layers, num_heads=num_heads, rep=rep, num_batches=num_batches)
            write_heads(os.path.join(model_dir, checkpoint), output_dir, head_type, scores, revision, threshold)

    elif os.path.isdir(model_dir):  # if one custom model
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                            num_layers=num_layers, num_heads=num_heads, rep=rep, num_batches=num_batches)
        write_heads(model_dir, output_dir, head_type, scores, revision, threshold)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if "gpt" in model_dir:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(model_dir)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = GPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
            scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                num_layers=num_layers, num_heads=num_heads, rep=rep, num_batches=num_batches)
            write_heads(model_dir, output_dir, head_type, scores, revision, threshold)
        elif 'pythia' in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = GPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
                scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                    num_layers=num_layers, num_heads=num_heads, rep=rep, num_batches=num_batches)
                write_heads(model_dir, output_dir, head_type, scores, revision, threshold)


if __name__ == "__main__":
    main()