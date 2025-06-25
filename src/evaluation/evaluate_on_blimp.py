"""
Author: Tatsuya

"""
import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from transformers import GPT2TokenizerFast, AutoTokenizer, DataCollatorWithPadding
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
import transformers, torch, argparse, pathlib, tqdm, json
from datasets import Dataset
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

def _get_data(data_dir, tokenizer):
    blimp_fns = os.listdir(data_dir)
    blimp_fns.sort()  # sort alphabetically so the file name can be inferred from id
    good, bad = [], []
    for fn in blimp_fns:
        with open(os.path.join(data_dir, fn)) as f:
            for line in f.readlines():  # no sorting here so other properties can be inferred by the original order
                dct = json.loads(line.strip())
                good.append(tokenizer.eos_token+dct['sentence_good'])
                bad.append(tokenizer.eos_token+dct['sentence_bad'])
    return good, bad

def _tokenize_and_batchify(sents, tokenizer, batch_size):
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_dict({'text': sents})
    def _tokenize(texts):
        return tokenizer(texts['text'])
    tokenized = dataset.map(_tokenize, batched=True)  # batched does not affect output
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])  # text column spits an error so remove it
    dataloader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)
    return dataloader

def _get_sentence_log_probas(model, batches, ablation_mode):

    model.eval()
    model.to(device)

    softmax = torch.nn.Softmax(dim=-1)

    results = []

    for batch in tqdm.tqdm(batches):
        batch.to(device)
        with torch.no_grad():
            outputs = model.forward_plus(ablation_mode=ablation_mode, **batch)
        probas = softmax(outputs.logits)  # batchsize, sequence length, vocab size
        shifted_target_ids = batch["input_ids"][:, 1:].contiguous()  # Shape: (batch_size, seq_length-1)
        shifted_attention_mask = batch["attention_mask"][:, 1:].contiguous()  # Shape: (batch_size, seq_length-1)
        target_probas = probas.gather(dim=-1, index=shifted_target_ids.unsqueeze(-1)).squeeze(-1)
        padded_target_probas = target_probas * shifted_attention_mask
        sent_probas = torch.log(padded_target_probas+1e-9).sum(dim=-1)  # (batchsize, max sequence length) so sum on the last dim
        results.extend(sent_probas.tolist())
    return results

def evaluate_on_blimp(data_dir, model, batch_size, tokenizer, ablation_mode):
    good, bad = _get_data(data_dir, tokenizer)
    good_batches = _tokenize_and_batchify(good, tokenizer, batch_size=batch_size)
    bad_batches = _tokenize_and_batchify(bad, tokenizer, batch_size=batch_size)
    good_probas = _get_sentence_log_probas(model, good_batches, ablation_mode)
    bad_probas = _get_sentence_log_probas(model, bad_batches, ablation_mode)
    # return sum([p1 > p2 for p1, p2 in zip(good_probas, bad_probas)]) / len(good_probas)
    return ''.join([str(int(p1 > p2)) for p1, p2 in zip(good_probas, bad_probas)])

def write_scores(model_dir, output_dir, score, revision, ablation_mode, ablation_head):
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    if ablation_mode:  # if ablation
        if ablation_head:
            attention_info = ablation_head  # e.g. surp@1-1-full (1-th layer, 1-th head, full ablation)
        attention_info += f'-{ablation_mode}'
        suffix = f'blimp@{attention_info}'
    else:  # no ablation
        suffix = 'blimp'
    # with open('test.tsv', 'w') as f:
    #     f.write(str(score))
    with open(os.path.join(output_dir, '-'.join([
        model_name, suffix])) + '.tsv', 'w') as f:
        f.write(str(score))
    correct = sum([int(i) for i in score])
    total = len(score)
    print(f"{model_name} {suffix} ||| {correct}/{total} ({round(correct/total, 3)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default=os.path.join(DATA_DIR, 'blimp'),
                        help=f"path to token file, default={os.path.join(DATA_DIR, 'blimp')}")
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help=f'batch size, default=512')
    parser.add_argument('-am', '--ablation_mode', choices=['full', 'pp'], default=None,
                        help="type of ablation to perform: ['full', 'pp'], default=None")
    parser.add_argument('-ah', '--ablation_head', default=None,
                        help="head to ablate e.g. '-ah 0-3',  default=None")
    parser.add_argument('-r', '--revision', required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'blimp_scores'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'blimp_scores')}")
    args = parser.parse_args()
    data_dir, model_dir, batch_size, ablation_mode, ablation_head, output_dir, revision = \
        args.data_dir, args.model_dir, args.batch_size, args.ablation_mode,\
        args.ablation_head, args.output_dir, args.revision
    if ablation_head:
        output_dir = os.path.join(DATA_DIR, 'by_head_ablation', 'blimp_scores')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # convert induction head
    if ablation_mode:
        l, h = ablation_head.split('-')
        ablation_head_dct = {int(l): [int(h)]}
    else:
        ablation_head_dct = []

    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(model_dir, checkpoint))
            model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint),
                                                            ablation_head_idx=ablation_head_dct)
            score = evaluate_on_blimp(data_dir=data_dir, model=model, tokenizer=tokenizer,
                                      batch_size=batch_size, ablation_mode=ablation_mode)
            write_scores(os.path.join(model_dir, checkpoint), output_dir, score, revision, ablation_mode, ablation_head)

    elif os.path.isdir(model_dir):  # if one custom model
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = AblationGPT2LMHeadModel.from_pretrained(model_dir, ablation_head_idx=ablation_head_dct)
        score = evaluate_on_blimp(data_dir=data_dir, model=model, tokenizer=tokenizer,
                                  batch_size=batch_size, ablation_mode=ablation_mode)
        write_scores(model_dir, output_dir, score, revision, ablation_mode, ablation_head)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if "gpt" in model_dir:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
                model = AblationGPT2LMHeadModel.from_pretrained(model_dir, ablation_head_idx=ablation_head_dct)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=ablation_head_dct)
            score = evaluate_on_blimp(data_dir=data_dir, model=model, tokenizer=tokenizer,
                                      batch_size=batch_size, ablation_mode=ablation_mode)
            write_scores(model_dir, output_dir, score, revision, ablation_mode, ablation_head)

        elif "pythia" in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=ablation_head_dct)
                score = evaluate_on_blimp(data_dir=data_dir, model=model, tokenizer=tokenizer,
                                          batch_size=batch_size, ablation_mode=ablation_mode)
                write_scores(model_dir, output_dir, score, revision, ablation_mode, ablation_head)

if __name__ == "__main__":
    main()