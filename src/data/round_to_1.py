import tqdm
pbar = tqdm.tqdm(total=1_000_000_000)
num_toks = 0
with open('data/cc100_1e_tokens_heads.jsonl', 'w') as ouf:
    with open('data/cc100_1b_tokens_heads.jsonl') as inf:
        for line in inf:
            ouf.write(line)
            num_toks += 1024
            pbar.update(1024)
            if num_toks >= 1_000_000_000:
                break