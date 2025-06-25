import os, pathlib, pickle, argparse
# os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
from transformers import GPT2TokenizerFast, AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', default='EleutherAI/pythia-70m',
                    help='tokenizer, default=EleutherAI/pythia-70m')
parser.add_argument('-d', '--data', default='EleutherAI/the_pile_deduplicated', # 'EleutherAI/the_pile'
                    help='tokenizer, default=EleutherAI/the_pile_deduplicated')
parser.add_argument('-t', '--train', action='store_true',
                    help='whether or not to create train')
parser.add_argument('-v', '--val', action='store_true',
                    help='whether or not to create val')
parser.add_argument('-hv', '--has_val', action='store_true',
                    help='data has val split')
parser.add_argument('-o', '--output_fn',
                    help='output filename')
args = parser.parse_args()

def tokenize(data, tokenizer, size, ctx_size, output_fn, num_files=10):
    file_ids = ['0' + str(i+1) if i < 9 else str(i+1) for i in range(num_files)]
    file_id, num_chunks, total_chunks = 0, 0, 0
    chunks, overflows = [], []
    target_num_chunks = int(size//ctx_size) if not size%ctx_size else int(size//ctx_size)+1
    perfile_num_chunks = int(target_num_chunks//num_files)+1
    pbar = tqdm(total=target_num_chunks)
    num_docs = 0
    for sample in data:
        num_docs += 1
        text = sample['text'].strip('\n').strip()
        print(f"\roverflows: {len(overflows)}\tchunks: {len(chunks)}\ttext: {len(text)}", end='')
        if not text:
            continue

        outputs = tokenizer(
            text,
            truncation=False
        )
        overflows.extend(outputs['input_ids'])
        curr_num_chunks = 0
        for i in range(0, len(overflows), ctx_size):
            chunk = overflows[i:i+ctx_size]
            if len(chunk) != ctx_size:  # last chunk
                overflows = chunk
                break
            chunks.append(chunk)
            curr_num_chunks += 1
        num_chunks += curr_num_chunks
        pbar.update(curr_num_chunks)
        while num_chunks >= min(target_num_chunks-total_chunks, perfile_num_chunks):  # make sure the total is at least the specified size
            with open(os.path.join(DATA_DIR, output_fn+'-'+file_ids[file_id]+'.pkl'), 'wb') as f:
                pickle.dump(chunks[:min(target_num_chunks-total_chunks, perfile_num_chunks)], f)
            file_id += 1
            total_chunks += min(target_num_chunks-total_chunks, perfile_num_chunks)
            del chunks[:min(target_num_chunks-total_chunks, perfile_num_chunks)]
            num_chunks -= min(target_num_chunks-total_chunks, perfile_num_chunks)
            if file_id == num_files:
                break
        if total_chunks >= target_num_chunks:
            break
        overflows.append(tokenizer.eos_token_id)
    return num_docs

def load_batches_from_pickles(dir, fns):
    for fn in fns:
        with open(os.path.join(dir, fn), 'rb') as file:
            # Load the entire list of batches from the pickle file
            batches = pickle.load(file)
            for batch in batches:
                yield batch  # Yield each batch

def main():
    train_size = 10_000_000_000
    val_size = 10_000_000
    # train_size = 1_000_000
    # val_size = 100_000
    ctx_size = 1_024
    data = load_dataset(args.data, streaming=True)
    if 'gpt' in args.model:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    elif 'pythia' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.has_val:
        data = data['validation']
        print('working on validation set...')
        num_docs = tokenize(data, tokenizer, val_size, ctx_size, args.output_fn, num_files=1)
        return

    data = data['train']



    if args.train:
        """TRAIN"""
        # tokenize
        print('working on training set...')
        num_docs = tokenize(data, tokenizer, train_size, ctx_size, args.output_fn, num_files=10)
        # chunk and save
        # chunk_and_save(train_toks, train_size, ctx_size, 'pile_10b_tokens.pkl')
    elif not args.train and args.val:
        num_docs = 0
        eos = GPT2TokenizerFast.from_pretrained('gpt2').eos_token_id
        pickles = sorted([fn for fn in os.listdir(DATA_DIR) if 'pile' in fn and '10b' in fn],
                         key=lambda x: int(x.split('.')[0].split('-')[1]))
        g = load_batches_from_pickles(DATA_DIR, pickles)
        pbar = tqdm(total=int(train_size//ctx_size)+1)
        for batch in g:
            print(f"\rcounting the number of documents in train: {num_docs}", end='')
            assert len(batch) == ctx_size
            num_docs += batch.count(eos)
            pbar.update()
        print(f"{num_docs} training documents found!", end='')

    """VAL"""
    # skip training part
    print('skipping the portion used for training set...')
    data = data.skip(num_docs)
    # tokenize
    print('working on validation set...')
    num_docs = tokenize(data, tokenizer, val_size, ctx_size, args.output_fn, num_files=1)
    # chunk and save
    # chunk_and_save(val_toks, val_size, ctx_size, 'pile_10m_tokens.pkl')

if __name__ == "__main__":
    main()
#
#

# text = tokenizer.decode(batches[3])
