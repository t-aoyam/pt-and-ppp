from transformers import AutoTokenizer
import torch, random, json, tqdm, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_toks', required=True, default=1_000_000_000, type=int,
                    help="number of total tokens, default=1B")
parser.add_argument('-l', '--low', required=True, default=1_000_000_000, type=int,
                    help="shortest random sequnence, default=50")
parser.add_argument('-h', '--high', required=True, default=512, type=int,
                    help="largest random sequence, default=512")
parser.add_argument('-c', '--ctx_size', required=True, default=1_024, type=int,
                    help="length of each sequence, default=1_024")

args = parser.parse_args()
target_num_toks, low, high, ctx_size = args.num_toks, args.low, args.high, args.ctx_size
target_num_seqs = (target_num_toks//1024 + 1)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
pbar = tqdm.tqdm(total=target_num_seqs)
with open(os.path.join('data', 'rep_seqs_1b.jsonl'), 'w') as f:
    for i in range(target_num_seqs):
        l = random.randint(low, high)
        input_tensor = torch.randint(0, len(dict(tokenizer.get_vocab())), (l,))
        input_tensor = input_tensor.repeat(1024//l+1)[:1024].tolist()

        base = (([1]+[0]*(l-1))*(ctx_size//l+1))[:1024]
        # indices = [
        #     ([0]*((src+1)%l)+base)[:ctx_size]
        #     for src in range(ctx_size)
        # ]

        d = {
            'input_ids': input_tensor,
            'indices': base,
            'span_length': l
        }
        f.write(json.dumps(d)+'\n')
        pbar.update()


"""test
for i, t in enumerate(input_tensors):
    l = span_lenghts[i]
    t = input_tensors[i]
    for j in range(len(t)):
        repeats = [index for index in range(j, 0, -l)]
        if repeats:
            print(repeats)
"""