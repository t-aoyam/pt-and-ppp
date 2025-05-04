# from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os
# os.environ['STANZA_RESOURCES_DIR'] = '/home/tatsuya/data/stanza_resources/'
# os.environ['SPACY_HOME'] = '/home/tatsuya/data/stanza_resources/'
import pathlib, tqdm#, torch
import spacy
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(f"cuda available: {torch.cuda.is_available()}")
# print(f"number of gpus: {torch.cuda.device_count()}")

nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer"])

ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
# doc = nlp('Hi how are you? My name is John .')
# print(doc.text)

data_size_for_lm = 10_000
context_length = 1_024
val_size = 1_000

# data_size_for_lm = 1_000_000_000
# context_length = 1_024
# val_size = 10_000_000

# target_num_words = 800_000_000  # should add up to 1b tokens
target_num_words = 800_000_000  # should add up to 1b tokens
#target_num_words = 80_000  # should add up to 1b tokens

# nlp = stanza.Pipeline(
#     lang='en',
#     processors='pos,tokenize,lemma,depparse',  # Only keep necessary processors
#     use_gpu=torch.cuda.is_available()  # Use GPU if available
# )
#
# tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
#
# i=0
# samples = []
# for sample in data:
#     samples.append(sample)
#     i += 1
#     if i == 10:
#         break
#
# for sample in samples:
#     print(sample['text'])
#
# test = nlp(samples[0]['text'])
def dep_parse_cc100():
    data = load_dataset('cc100', 'en', streaming=True)
    data = data['train']
    num_words = 0
    conllu = ['# newdoc']
    pbar = tqdm.tqdm(total=target_num_words)
    with open(os.path.join(DATA_DIR, 'cc100_1b_tokens.conllu'), 'a') as f:
        for sample in data:
            text = preprocess(sample)
            if not text:
                conllu.append('# newdoc')
                continue

            conllu.append('# text = ' + text)
            nlped = nlp(text)
            offset = -1
            for sent in nlped.sents:
                pbar.update(len(sent))
                num_words += len(sent)
                conllu.append('# newsent')
                for token in sent:
                    conllu.append('\t'.join(
                        [str(token.i - offset), token.text, str(token.head.i - offset), token.dep_]
                    ))
                offset += len(sent)
            f.write('\n'.join(conllu))
            conllu = ['']  # this will add '\n' at the beginning when joined
            if num_words >= target_num_words:
                print('done')
                break

    return None

def preprocess(sample):
    text = sample['text'].strip('\n').strip()
    if not text or max([len(word) for word in text.split()]) > 150 or len(text) > 1_000_000:
        return None
    # text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f]', ' ', text)
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f\v\x00-\x1F]', ' ', text)
    text = re.sub(r' +', ' ', text)
    if not text:
        return None
    return text

def dep_parse_pile():
    data = load_dataset('EleutherAI/the_pile_deduplicated', streaming=True)
    data = data['train']
    num_words = 0
    conllu = ['# newdoc']
    pbar = tqdm.tqdm(total=target_num_words)
    with open(os.path.join(DATA_DIR, 'pile_1b_tokens.conllu'), 'w') as f:
        for sample in data:
            text = preprocess(sample)
            if not text:
                continue
            conllu.append('# text = ' + text)
            nlped = nlp(text)
            offset = -1
            for sent in nlped.sents:
                pbar.update(len(sent))
                num_words += len(sent)
                conllu.append('# newsent')
                for token in sent:
                    conllu.append('\t'.join(
                        [str(token.i - offset), token.text, str(token.head.i - offset), token.dep_]
                    ))
                offset += len(sent)
            f.write('\n'.join(conllu))
            if num_words >= target_num_words:
                print('done')
                break
            conllu = ['', '# newdoc']  # this will add '\n' at the beginning when joined

    return None

# dep_parse_pile()

dep_parse_cc100()