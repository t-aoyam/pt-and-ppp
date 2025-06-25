import os, json, tqdm
pbar = tqdm.tqdm(total=1_000_000_000//1_024+1)
with open(os.path.join('data', 'rep_seqs_1b.jsonl'), 'w') as ouf:
    with open(os.path.join('data', 'rep_seqs_1b_with_bases.jsonl')) as inf:
        for line in inf:
            d = json.loads(line.strip())
            del d['indices']
            ouf.write(json.dumps(d)+'\n')
            pbar.update()