import os
# os.environ['STANZA_RESOURCES_DIR'] = '/home/tatsuya/data/stanza_resources/'
# os.environ['SPACY_HOME'] = '/home/tatsuya/data/stanza_resources/'
import tqdm


def open_as_generator(conllu_fp):
    with open(conllu_fp) as f:
        for line in f:
            yield line

pbar = tqdm.tqdm(total=800_000_000)
with open(os.path.join('data', 'pile_1b_tokens_cleaned.conllu'), 'w') as fi:
    for line in open_as_generator(os.path.join('data', 'pile_1b_tokens.conllu')):
        if line.startswith('#'):
            fi.write(line)
        elif len(line.strip().split('\t')) == 4:
            fi.write(line)
        pbar.update()

for line in open_as_generator(os.path.join('data', 'pile_1b_tokens_cleaned.conllu')):
    if line.startswith('#'):
        continue
    if line.split('\t')
    elif len(line.split('\t')[1]) > 150:
        print(line.split('\t'))
