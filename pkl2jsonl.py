import json, pickle, tqdm, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True,
                    help='pkl file')
args = parser.parse_args()
fp = args.file

def main():
    with open(fp, 'rb') as f:
        ids = pickle.load(f)

    pbar = tqdm.tqdm(total=1_000_000_000)
    fn = fp.split(os.path.sep)[-1].split('.')[0]
    with open(fn+'.'+'jsonl', 'w') as f:
        for start_idx in range(0, 1_000_001_024, 1024):
            chunk = ids[start_idx:start_idx + 1024]
            if len(chunk) < 1024:
                break
            f.write(json.dumps({'input_ids': chunk})+'\n')
            pbar.update(1024)


if __name__ == "__main__":
    main()