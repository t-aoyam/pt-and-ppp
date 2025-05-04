"""
Author: Tatsuya
borrowed from: https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_llm_surprisal.py

"Token indices sequence length is longer than the specified maximum sequence length for this model
 (1289 > 1024). Running this sequence through the model will result in indexing errors"
 -> this warning is a friendly reminder at the time of tokenization, so can be safely ignored

Take word-code .tsv file (all_toks.tsv) and assign an LM surprisal.
Writes 1 .tsv:
modelname_surps.tsv: each row contains a code-word-surp pair.
modelname looks like: gpt2-type-[num_layer]-layers-checkpoint-[checkpoint]
---
corpus-storyID-tokenID    word  surp
dundee-1-1  Abc 20.11
dundee-1-2  def 12.31
...
ns-10-1000  xyz ???
---
"""
import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from transformers import AutoTokenizer, AutoConfig
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
import torch, argparse, pathlib, tqdm, re
import numpy as np
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

def get_data(data_fp, split_stories=True):
    data = [line.strip().split('\t') for line in open(data_fp).readlines() if line.strip()][:100000]
    wid2gid = {line[0]:i for i, line in enumerate(data)}
    gid2info = {i: line for i, line in enumerate(data)}
    if not split_stories:
        return [line[1] for line in data], [line[0] for line in data], gid2info, wid2gid
    stories, codes, curr_story_idx = [], [], data[0][0].split('-')[0]
    story_codes, story_toks = [], []
    for code, tok, _, _, _, _, _ in data:
        story_idx, tok_idx = code.split('-')
        if story_idx != curr_story_idx:
            assert len(story_codes) == len(story_toks), ValueError("Numbers of tokens and codes don't match")
            stories.append(story_toks)
            codes.append(story_codes)
            story_codes, story_toks, curr_story_idx = [], [], story_idx
        story_toks.append(tok)
        story_codes.append(code)
    # last story
    assert len(story_codes) == len(story_toks), ValueError("Numbers of tokens and codes don't match")
    stories.append(story_toks)
    codes.append(story_codes)

    return stories, codes, gid2info, wid2gid


def _tokenize_and_batchify(tokenizer, ctx_size, stride, batch_size, words):
    # docs, codes = get_data(os.path.join(DATA_DIR, 'ewt.txt'))
    concat = ' '.join(words)
    concat = re.sub(r" <\|endoftext\|> ", r"<|endoftext|>", concat)
    toks = tokenizer(concat)
    # create token to word mapping
    curr_toks = []
    word_id = 0
    tok2word = dict()
    for i, tok in enumerate(toks['input_ids']):
        curr_toks.append(tok)
        if tokenizer.decode(curr_toks).strip() == words[word_id]:
            for tok_id in range(i-len(curr_toks)+1, i+1):
                tok2word[tok_id] = word_id
            assert tokenizer.decode(toks['input_ids'][i-len(curr_toks)+1:i+1]).strip() == words[word_id]
            # print(tokenizer.decode(toks['input_ids'][i-len(curr_toks)+1:i+1]).strip(), words[word_id])
            word_id += 1
            curr_toks = []
    input_ids = toks['input_ids']
    batch = []
    batches = []
    for i in range(0, len(input_ids), stride):
        batch.append(input_ids[i:i+ctx_size])
        if len(batch[-1]) < ctx_size:
            break
        if len(batch) == batch_size:
            batches.append(torch.tensor(batch))
            batch = []
    # batches.append(torch.tensor(batch))
    return batches, tok2word, toks

def get_head2score(model_dir, model, tokenizer, ctx_size, stride, batch_size, words, gid2info, wid2gid):

    model.eval()
    model.to(device)

    config = AutoConfig.from_pretrained(model_dir)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    batches, tok2word, toks = _tokenize_and_batchify(tokenizer, ctx_size, stride, batch_size, words)

    pbar = tqdm.tqdm(total=len(batches))
    head2score = {'-'.join([str(l), str(h)]): {'i': [[], []], 'abs_p': [[], []],
                                               'rel_p': [[], []], 's': [[], []]} \
                  for l in range(num_layers) \
                  for h in range(num_heads)}
    with torch.no_grad():
        for batch_id, batch in enumerate(batches):
            output = model(batch.to(device), output_attentions=True)
            for seq_id in range(batch_size):
                for layer_id in range(num_layers):
                    for head_id in range(num_heads):
                        for source_id in range(stride, ctx_size):
                            # all previous batches
                            cml_tok_id = batch_id * (ctx_size + stride * (batch_size - 2)) + \
                                         stride * seq_id  # all previous sequences in the current batch
                            cml_tok_id = int(cml_tok_id)
                            if int(batches[batch_id][seq_id][source_id]) == tokenizer.eos_token_id:
                                continue

                            assert int(batches[batch_id][seq_id][source_id]) == toks['input_ids'][cml_tok_id + source_id]

                            head = '-'.join([str(layer_id), str(head_id)])
                            max_attn = torch.max(output.attentions[layer_id][seq_id][head_id][source_id])
                            target_id = int(torch.argmax(output.attentions[layer_id][seq_id][head_id][source_id]))

                            # convert to word level info
                            target_word_id = tok2word[cml_tok_id + target_id]
                            source_word_id = tok2word[cml_tok_id + source_id]

                            head2score[head]['abs_p'][0].append(target_id)  # position of the most attended token
                            head2score[head]['abs_p'][1].append(max_attn)  # its confidence ~ attention weight

                            head2score[head]['rel_p'][0].append(
                                target_id - source_id)  # position of the most attended token
                            head2score[head]['rel_p'][1].append(max_attn)  # its confidence ~ attention weight

                            parent = [wid2gid[gid2info[source_word_id][3]]] \
                                if '--' not in gid2info[source_word_id][3] and gid2info[source_word_id][
                                3] != '<|endoftext|>' else []
                            children = [wid2gid[idx] for idx in [
                                child.split(':')[0] for child in gid2info[source_word_id][5].split(',')
                            ]] if gid2info[source_word_id][5] else []
                            is_dep = target_word_id in parent or target_word_id in children

                            head2score[head]['s'][0].append(int(is_dep))
                            head2score[head]['s'][1].append(max_attn)

                            head2score[head]['i'][0].append(gid2info[target_word_id][6])
                            head2score[head]['i'][1].append(max_attn)
            pbar.update(1)

    return head2score


def write_heads(model_dir, output_dir, head2score, revision):
    out = []
    for head in head2score:
        s_ratio = np.mean([float(d) for d in head2score[head]['s'][0]])
        w_freq = np.mean([float(d) for d in head2score[head]['i'][0]])
        absp_pos, absp_freq = sorted(list(Counter(head2score[head]['abs_p'][0]).items()), key=lambda x:x[1], reverse=True)[0]
        relp_pos, relp_freq = sorted(list(Counter(head2score[head]['rel_p'][0]).items()), key=lambda x:x[1], reverse=True)[0]
        total = len(head2score[head]['s'][0])
        conf = np.mean([float(d) for d in head2score[head]['s'][1]])
        out.append([head, s_ratio, w_freq, absp_pos, absp_freq, relp_pos, relp_freq, total, conf])

    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    with open(os.path.join(output_dir, '-'.join([model_name, f'other_heads'])) + '.tsv', 'w') as f:
        f.write('\n'.join(
            ['\t'.join([str(num) for num in line]) for line in out]
        )
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fp', default=os.path.join(DATA_DIR, 'ewt.txt'),
                        help=f"path to token file, default={os.path.join(DATA_DIR, 'ewt.txt')}")
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-r', '--revision', required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-c', '--ctx_size', default=1024,
                        help=f'context LMs will use, default=max=1024')
    parser.add_argument('-s', '--stride', default=None,
                        help=f'stride for moving window, default=context/2')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help=f'batch size, default=8')
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'other_heads'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'other_heads')}")
    args = parser.parse_args()

    data_fp, model_dir, ctx_size, stride, output_dir, revision, batch_size = \
        args.data_fp, args.model_dir,\
        int(args.ctx_size), args.stride, args.output_dir, args.revision, args.batch_size

    stride = int(ctx_size/2) if not stride else int(stride)  # stride defaults to 50% overlap sliding window
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # get data
    words, codes, gid2info, wid2gid = get_data(os.path.join(DATA_DIR, 'ewt.txt'), split_stories=False)

    # load tokenizer and model
    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, checkpoint))
            model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint))
            head2score = get_head2score(model_dir=model_dir, model=model, tokenizer=tokenizer,
                                        ctx_size=ctx_size, stride=stride, batch_size=batch_size,
                                        words=words, gid2info=gid2info, wid2gid=wid2gid)
            write_heads(os.path.join(model_dir, checkpoint), output_dir, head2score, revision)

    elif os.path.isdir(model_dir):  # if one custom model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AblationGPT2LMHeadModel.from_pretrained(model_dir)
        head2score = get_head2score(model_dir=model_dir, model=model, tokenizer=tokenizer,
                                    ctx_size=ctx_size, stride=stride, batch_size=batch_size,
                                    words=words, gid2info=gid2info, wid2gid=wid2gid)
        write_heads(model_dir, output_dir, head2score, revision)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if "gpt" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AblationGPT2LMHeadModel.from_pretrained(model_dir)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}')
            head2score = get_head2score(model_dir=model_dir, model=model, tokenizer=tokenizer,
                                        ctx_size=ctx_size, stride=stride, batch_size=batch_size,
                                        words=words, gid2info=gid2info, wid2gid=wid2gid)
            write_heads(model_dir, output_dir, head2score, revision)

        elif "pythia" in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}')
                head2score = get_head2score(model_dir=model_dir, model=model, tokenizer=tokenizer,
                                            ctx_size=ctx_size, stride=stride, batch_size=batch_size,
                                            words=words, gid2info=gid2info, wid2gid=wid2gid)
                write_heads(model_dir, output_dir, head2score, revision)


if __name__ == "__main__":
    main()