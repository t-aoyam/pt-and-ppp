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
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, GPTNeoXForCausalLM, AutoTokenizer
from ablated_gpt2 import AblationGPT2LMHeadModel
from ablated_pythia import AblationGPTNeoXForCausalLM
import transformers, torch, argparse, pathlib, tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

def get_data(data_fp, split_stories=True):
    data = [line.strip().split('\t') for line in open(data_fp).readlines() if line.strip()]
    if not split_stories:
        return data
    stories, codes, curr_story_idx = [], [], '-'.join(data[0][0].split('-')[:2])
    story_codes, story_toks = [], []
    for code, tok in data:
        corpus, story_idx, tok_idx = code.split('-')
        story_idx = '-'.join([corpus, story_idx])
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

    return stories, codes

def get_surprisal(unit, model, ablation_mode, tokenizer, ctx_size, stride, stories, verbose=False):
    assert unit in {"token", "word"}, ValueError('Calculation unit must be "token" or "word"')

    model.eval()
    model.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    bos_id = model.config.eos_token_id

    batches = []
    words = []
    for story_words in stories:
        words.extend(story_words)
        story = ' '.join(story_words)
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default (GPT and Pythia)
        ids = [bos_id] + ids
        attn = [1] + attn

        start_idx = 0

        # sliding windows with 50% overlap by default, or specified stride
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # for models that explicitly require the first dimension (batch_size)
            if "gpt-neox" in model.config._name_or_path or "pythia" in model.config._name_or_path:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                            "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                torch.tensor(ids[1:ctx_size+1]),
                                start_idx,
                                True))
            # for other models
            elif "gpt" in model.config._name_or_path:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]),
                                                            "attention_mask": torch.tensor(attn[:ctx_size])}),
                                torch.tensor(ids[1:ctx_size+1]),
                                start_idx,
                                True))

            ids = ids[int(stride):]
            attn = attn[int(stride):]
            start_idx = int(stride)

        # remaining tokens
        if "gpt-neox" in model.config._name_or_path or "pythia" in model.config._name_or_path:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn[:-1]).unsqueeze(0)}),
                           torch.tensor(ids[1:]),
                            start_idx,
                            False))
        elif "gpt" in model.config._name_or_path:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]),
                                                        "attention_mask": torch.tensor(attn[:-1])}),
                            torch.tensor(ids[1:]),
                            start_idx,
                            False))

    # print("word llmsurp")
    curr_word_surp = []
    curr_toks = []
    curr_word_ix = 0
    is_continued = False
    surps = []
    pbar = tqdm.tqdm(total=len(stories))
    for batch in batches:
        batch_input, output_ids, start_idx, will_continue = batch
        batch_input.to(device)
        with torch.no_grad():
            model_output = model.forward_plus(ablation_mode=ablation_mode, **batch_input)

        toks = tokenizer.convert_ids_to_tokens(output_ids)
        index = torch.arange(0, output_ids.shape[0])
        surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])

        if unit == "token":
            # token-level surprisal
            for i in range(start_idx, len(toks)):
                cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                if verbose:
                    print(cleaned_tok, surp[i].item())
                surps.append([cleaned_tok, surp[i].item()])

        elif unit == "word":
            # word-level surprisal
            # if the batch starts a new story
            if not is_continued:
                pbar.update(1)
                curr_word_surp = []
                curr_toks = []
                curr_positions = []
            for i in range(start_idx, len(toks)):
                curr_word_surp.append(surp[i].item())
                curr_positions.append(i)
                curr_toks += [toks[i]]
                curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
                # summing token-level surprisal
                if words[curr_word_ix] == curr_toks_str.strip():
                    if verbose:
                        print(curr_toks_str.strip(), sum(curr_word_surp))
                    surps.append([curr_toks_str.strip(), sum(curr_word_surp), curr_positions])
                    curr_word_surp = []
                    curr_positions = []
                    curr_toks = []
                    curr_word_ix += 1

        is_continued = will_continue
        del model_output

    if unit == 'word':
        assert [tok.lower() for tok, _, _ in surps] == [tok.lower()
                                                       for story in stories
                                                       for tok in story],\
        ValueError("Numbers of words and surprisals don't match")

    return ([surp for _, surp, _ in surps], [p for _, _, p in surps])

def write_surps(model_dir, output_dir, surps, stories, codes, positions, ctx_size, stride,
                revision, ablation_mode, ablation_threshold, ablation_head):
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    if ablation_mode:  # if ablation
        if ablation_threshold:
            attention_info = str(int(100 * ablation_threshold))  # e.g. surp@20-full
        elif ablation_head:
            attention_info = ablation_head  # e.g. surp@1-1-full (1-th layer, 1-th head, full ablation)
        attention_info += f'-{ablation_mode}'
        suffix = f'surps@{attention_info}'
    else:  # no ablation
        suffix = 'surps'

    out = []
    for code, tok, surp, position in zip(
            [code for story_codes in codes for code in story_codes],
            [tok for story in stories for tok in story],
            surps,
            positions):
        out.append('\t'.join([code, tok, str(surp), '-'.join([str(p) for p in position])]))

    with open(os.path.join(output_dir, '-'.join([
        model_name, str(ctx_size), str(int(stride)),
                suffix])) + '.tsv', 'w') as f:
        f.write('\n'.join(out))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unit', default='word',
                        help="'token' or 'word', default='word'")
    parser.add_argument('-d', '--data_fp', default=os.path.join(DATA_DIR, 'rt_data', 'all_toks.tsv'),
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
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'surps'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'surps')}")
    args = parser.parse_args()
    unit, data_fp, model_dir, ablation_mode, ablation_threshold, ablation_head, ctx_size, stride, output_dir, revision = \
        args.unit, args.data_fp, args.model_dir, args.ablation_mode, args.ablation_threshold, args.ablation_head,\
        int(args.ctx_size), args.stride, args.output_dir, args.revision
    if ablation_threshold and ablation_head:
        raise IOError('Ablation heads must be specified by *either* pms threshold or head id!')
    if ablation_head:
        output_dir = os.path.join(DATA_DIR, 'by_head_ablation', 'surps')
    stride = int(ctx_size/2) if not stride else int(stride)  # stride defaults to 50% overlap sliding window
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # get data
    stories, codes = get_data(data_fp)
    # load tokenizer and model
    induction_heads = []  # initialize
    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(model_dir, checkpoint))
            if ablation_mode:
                induction_heads = induction_tsv2dct(os.path.join(model_dir, checkpoint), revision,
                                                    ablation_threshold, ablation_head)
            model = AblationGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint),
                                                            ablation_head_idx=induction_heads)
            surps, positions = get_surprisal(unit=unit, model=model, tokenizer=tokenizer,
                                             ctx_size=ctx_size, stride=stride, stories=stories,
                                             ablation_mode=ablation_mode)

            write_surps(os.path.join(model_dir, checkpoint), output_dir,
                        surps, stories, codes, positions, ctx_size, stride, revision,
                        ablation_mode, ablation_threshold, ablation_head)

    elif os.path.isdir(model_dir):  # if one custom model
        if ablation_mode:
            induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = AblationGPT2LMHeadModel.from_pretrained(model_dir, ablation_head_idx=induction_heads)
        surps, positions = get_surprisal(unit=unit, model=model, tokenizer=tokenizer,
                                         ctx_size=ctx_size, stride=stride, stories=stories,
                                         ablation_mode=ablation_mode)
        write_surps(model_dir, output_dir, surps, stories, codes, positions,
                    ctx_size, stride, revision, ablation_mode, ablation_threshold, ablation_head)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if ablation_mode:
                induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
            if "gpt" in model_dir:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
                model = AblationGPT2LMHeadModel.from_pretrained(model_dir, ablation_head_idx=induction_heads)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=induction_heads)
            surps, positions = get_surprisal(unit=unit, model=model, tokenizer=tokenizer,
                                             ctx_size=ctx_size, stride=stride, stories=stories,
                                             ablation_mode=ablation_mode)
            write_surps(model_dir, output_dir, surps, stories, codes, positions,
                        ctx_size, stride, revision, ablation_mode, ablation_threshold, ablation_head)

        elif "pythia" in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                if ablation_mode:
                    induction_heads = induction_tsv2dct(model_dir, revision, ablation_threshold, ablation_head)
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f'step{str(revision)}')
                model = AblationGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f'step{str(revision)}',
                                                                   ablation_head_idx=induction_heads)
                surps, positions = get_surprisal(unit=unit, model=model, tokenizer=tokenizer,
                                                 ctx_size=ctx_size, stride=stride, stories=stories,
                                                 ablation_mode=ablation_mode)
                write_surps(model_dir, output_dir, surps, stories, codes, positions,
                            ctx_size, stride, revision, ablation_mode, ablation_threshold, ablation_head)

if __name__ == "__main__":
    main()