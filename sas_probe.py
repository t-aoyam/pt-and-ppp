import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
import torch, pathlib, argparse, sys
from tqdm import tqdm
from word_level_input_output import WordLevelIO
from transformers import AutoTokenizer
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

def head_probe(wlio, scale=True):
    """
    x_i -> x_j attention and x_i <- x_j attention
    e.g.
    1       0       0       0       0
    0.3     0.7     0       0       0
    0.2     0.2     0.6     0       0
    0.1     0.1     0.5     0.4     0
    0.01    0.09    0.3     0.3     0.3
    -> probe says: maxes that go through the diagonal points are the parents

    1. scaling based on the number of 'attendable' tokens
    2. crude 1024 with 1024 stride vs sentence level
    """
    assert wlio.ctx_size == wlio.stride, 'Currently this probe only supports non-overlapping sliding window.'
    by_head_predictions = []
    num_words = 0
    pbar = tqdm(total=len(wlio.batched_attentions))
    for seq_id, seq in enumerate(wlio.batched_attentions):
        seq.to(wlio.device)
        first_tok_id = wlio.ctx_size * seq_id
        # if final token in the last seq and the first token in the current batch belong to the same word
        if wlio.tok2word[first_tok_id] == num_words-1:
            by_head_predictions = by_head_predictions[:-1]  # just use the current token
            num_words -= 1  # consider this boundary 'taken care of'
        if scale:
            scaling_factor = torch.tensor(list(range(1, seq.shape[-1]+1)), device=device)
            scaling_factor = scaling_factor.expand(*seq.shape)
            seq = seq * scaling_factor  # seq should have a shape (layer, head, source, target)
        both = torch.cat([seq, seq.permute(0, 1, 3, 2)], dim=-1)
        max_ids = torch.argmax(both, dim=-1)
        # source-target concatenated, so fix the id, and change seq-specific id to global id
        max_ids = [idx%seq.shape[-1]+num_words for idx in max_ids.flatten()]
        max_ids = torch.tensor(max_ids, device=device).reshape(*seq.shape[:-1])  # put it back to the original shape
        max_ids = max_ids.permute(2,0,1).view(seq.shape[-1], seq.shape[0], seq.shape[1])  # (word, layer, head)
        by_head_predictions.extend(max_ids.tolist())
        num_words += seq.shape[-1]
        pbar.update(1)
    # print(len(by_head_predictions), len(wlio.word2tok))
    assert len(by_head_predictions) == len(wlio.word2tok), 'Numbers of words do not match.'
    return by_head_predictions

def DEP_get_and_write_by_head_predictions(
        data_fp, model, tokenizer, ctx_size, stride, batch_size,
        scale, device, output_dir, model_dir, revision
):
    # get predictions
    # ['pred-pred-pred...-pred', 'pred-pred-pred-...pred', ]
    wlio = WordLevelIO(data_fp, tokenizer, ctx_size, stride, batch_size, device)
    wlio.get_attentions(model)
    predictions = head_probe(wlio, scale)
    out = '\n'.join([
        '\t'.join([
            '-'.join([str(idx) for idx in layer])
            for layer in word
        ])
        for word in predictions
    ])

    # write to a file
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    suffix = '@scale' if scale else '@unscale'
    with open(os.path.join(output_dir, '-'.join([model_name, f'sas_preds', suffix])) + '.tsv', 'w') as f:
        f.write(out)


def get_and_write_by_head_predictions(
        data_fp, model, tokenizer, ctx_size, stride, batch_size,
        scale, device, output_dir, model_dir, revision
):
    # get predictions
    # ['pred-pred-pred...-pred', 'pred-pred-pred-...pred', ]
    wlio = WordLevelIO(data_fp=data_fp, tokenizer=tokenizer, ctx_size=ctx_size,
                       stride=stride, batch_size=batch_size, device=device, scale=scale)
    predictions = wlio.get_sas_preds(model)
    out = '\n'.join([
        '\t'.join([
            '-'.join([str(idx) for idx in layer])
            for layer in word
        ])
        for word in predictions
    ])

    # write to a file
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    suffix = '@scaled' if scale else '@unscaled'
    with open(os.path.join(output_dir, '-'.join([model_name, f'sas_preds'])+suffix) + '.tsv', 'w') as f:
        f.write(out)
def main():
    ROOT_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fp', default=os.path.join(DATA_DIR, 'ewt.txt'),
                        help=f"path to token file, default={os.path.join(DATA_DIR, 'ewt.txt')}")
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-r', '--revision', required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-c', '--ctx_size', type=int, default=1024,
                        help=f'context LMs will use, default=max=1024')
    parser.add_argument('-s', '--stride', type=int, default=1024,
                        help=f'stride for moving window, default=1024')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help=f'batch size, default=4')
    parser.add_argument('-n', '--num_words', type=int, default=10_000,
                        help=f'number of words to process at a time, default=10_000')
    parser.add_argument('-u', '--unscale_weights', action='store_true',
                        help=f'unscale attention weights, default is false')
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'sas_preds'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'sas_preds')}")
    args = parser.parse_args()

    data_fp, model_dir, ctx_size, stride, output_dir, revision, batch_size, num_words, unscale = \
        args.data_fp, args.model_dir, args.ctx_size, args.stride,\
        args.output_dir, args.revision, args.batch_size, args.num_words, args.unscale_weights

    scale = unscale == 0
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            print(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, checkpoint))
            model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint))
            get_and_write_by_head_predictions(
                data_fp=data_fp, model=model, tokenizer=tokenizer, ctx_size=ctx_size, stride=stride,
                batch_size=batch_size, scale=scale, device=device, output_dir=output_dir,
                model_dir=os.path.join(model_dir, checkpoint), revision=revision)
    elif os.path.isdir(model_dir):  # if one custom model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AblationGPT2LMHeadModel.from_pretrained(model_dir)
        get_and_write_by_head_predictions(
            data_fp=data_fp, model=model, tokenizer=tokenizer, ctx_size=ctx_size,
            stride=stride, batch_size=batch_size, scale=scale, device=device,
            output_dir=output_dir, model_dir=model_dir, revision=revision
        )

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if "gpt" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AblationGPT2LMHeadModel.from_pretrained(model_dir)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
            get_and_write_by_head_predictions(
                data_fp=data_fp, model=model, tokenizer=tokenizer, ctx_size=ctx_size,
                stride=stride, batch_size=batch_size, scale=scale, device=device,
                output_dir=output_dir, model_dir=model_dir, revision=revision
            )
        elif 'pythia' in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
                get_and_write_by_head_predictions(
                    data_fp=data_fp, model=model, tokenizer=tokenizer, ctx_size=ctx_size,
                    stride=stride, batch_size=batch_size, scale=scale, device=device,
                    output_dir=output_dir, model_dir=model_dir, revision=revision
                )


if __name__ == "__main__":
    main()
"""TEST
data_fp = os.path.join(os.getcwd(), 'data', 'ewt.txt')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
model = AblationGPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
ctx_size = 1024
stride = 1024
batch_size = 2
device = 'cpu'
wlio = WordLevelIO(data_fp, tokenizer, ctx_size, stride, batch_size, device)
predictions = head_probe(wlio, model)
write = []
for word in predictions:
    write.append(
        '\t'.join(
            [
                '-'.join([str(idx) for idx in layer])
                for layer in word
            ]
        )
    )
# attentions = wlio.get_attentions(model, 'word')

# seq = attentions[0]
# both = torch.cat([seq, seq.permute(0,1,3,2)], dim=-1)
# max_ids = torch.argmax(both, dim=-1)
# max_ids.shape

# for idx in max_ids:
# for i, both in enumerate(zip(seq, seq.permute(0, 1, 3, 2))):
#     outgoing, incoming = both
#     idx = np.argmax(outgoing.tolist() + incoming.tolist())
#     if idx >= len(outgoing.tolist()):
#         row, col = idx - len(outgoing), i
#         parent = row
#     else:
#         row, col = i, idx
#         parent = col

# for i, both in enumerate(zip(seq, seq.permute(0, 1, 3, 2))):
#     outgoing, incoming = both
#     break
#     idx = np.argmax(outgoing.tolist() + incoming.tolist())
#     if idx >= len(outgoing.tolist()):
#         row, col = idx - len(outgoing), i
#         parent = row
#     else:
#         row, col = i, idx
#         parent = col
"""