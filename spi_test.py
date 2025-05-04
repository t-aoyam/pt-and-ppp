import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from transformers import AutoTokenizer, AutoConfig
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
import transformers, torch, argparse, pathlib, tqdm
import re
import numpy as np
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

model = AblationGPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
# tokenizer.eos_token

def get_data(data_fp, split_stories=True):
    data = [line.strip().split('\t') for line in open(data_fp).readlines() if line.strip()]
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

words, codes, gid2info, wid2gid = get_data(os.path.join(DATA_DIR, 'ewt.txt'), split_stories=False)


def tokenize_and_batchify(ctx_size, stride, batch_size, words):
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

config = AutoConfig.from_pretrained('EleutherAI/pythia-70m')
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads

ctx_size, stride, batch_size = 1024, 512, 4
batches, tok2word, toks = tokenize_and_batchify(ctx_size=ctx_size, stride=stride,
                                                batch_size=batch_size, words=words)

pbar = tqdm.tqdm(total=len(batches))
head2score = {'-'.join([str(l), str(h)]): {'i': [[], []], 'abs_p': [[], []],
                                           'rel_p': [[], []], 's': [[], []]}\
              for l in range(num_layers)\
              for h in range(num_heads)}

def get_head2score():
    for batch_id, batch in enumerate(batches):
        output = model(batch, output_attentions=True)
        for seq_id in range(batch_size):
            for layer_id in range(num_layers):
                for head_id in range(num_heads):
                    for source_id in range(stride, ctx_size):
                        # all previous batches
                        cml_tok_id = batch_id * (ctx_size + stride * (batch_size-2)) + \
                                     stride * seq_id  # all previous sequences in the current batch
                        if int(batches[batch_id][seq_id][source_id]) == tokenizer.eos_token_id:
                            continue

                        assert int(batches[batch_id][seq_id][source_id]) == toks['input_ids'][cml_tok_id+source_id]

                        head = '-'.join([str(layer_id), str(head_id)])
                        attention = output.attentions[layer_id][seq_id][head_id][source_id]
                        max_attn = max(output.attentions[layer_id][seq_id][head_id][source_id].detach().numpy())
                        target_id = np.argmax(output.attentions[layer_id][seq_id][head_id][source_id].detach().numpy())

                        # convert to word level info
                        target_word_id = tok2word[cml_tok_id + target_id]
                        source_word_id = tok2word[cml_tok_id + source_id]

                        head2score[head]['abs_p'][0].append(target_id)  # position of the most attended token
                        head2score[head]['abs_p'][1].append(max_attn)  # its confidence ~ attention weight

                        head2score[head]['rel_p'][0].append(target_id-source_id)  # position of the most attended token
                        head2score[head]['rel_p'][1].append(max_attn)  # its confidence ~ attention weight

                        parent = [wid2gid[gid2info[source_word_id][3]]]\
                            if '--' not in gid2info[source_word_id][3] and gid2info[source_word_id][3] != '<|endoftext|>' else []
                        children = [wid2gid[idx] for idx in [
                            child.split(':')[0] for child in gid2info[source_word_id][5].split(',')
                        ]] if gid2info[source_word_id][5] else []
                        is_dep = target_word_id in parent or target_word_id in children

                        head2score[head]['s'][0].append(int(is_dep))
                        head2score[head]['s'][1].append(max_attn)

                        head2score[head]['i'][0].append(gid2info[target_word_id][6])
                        head2score[head]['i'][1].append(max_attn)
        pbar.update(1)

def write(head2score):
    out = []
    for head in head2score:
        s_ratio = np.mean([float(d) for d in head2score[head]['s'][0]])
        w_freq = np.mean([float(d) for d in head2score[head]['i'][0]])
        absp_pos, absp_freq = sorted(list(Counter(head2score[head]['abs_p'][0]).items()), key=lambda x:x[1], reverse=True)[0]
        relp_pos, relp_freq = sorted(list(Counter(head2score[head]['rel_p'][0]).items()), key=lambda x:x[1], reverse=True)[0]
        total = len(head2score[head]['s'][0])
        conf = np.mean([float(d) for d in head2score[head]['s'][1]])
        out.append([head, s_ratio, w_freq, absp_pos, absp_freq, relp_pos, relp_freq, total, conf])
    return out
write(head2score)
# l-y, syntax-ratio, syntax-conf, word-freq, word-conf, absp-pos, absp-freq, absp-conf, relp-pos, relp-freq, relp-conf, total
# (layer, batch, head, source, target)