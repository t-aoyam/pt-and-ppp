"""
Integrated a separate conllu2json.py
-> no ' '.join(words), so the text is faithful to the original corpus (no extra whitespaces)
"""
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os
import pathlib, tqdm, spacy, re, json#, torch
from spacy.symbols import ORTH
from spacy.util import compile_infix_regex
import string
string.punctuation
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# tokenizer.vocab
from spacy.tokens import Doc

# Load SpaCy pipeline and GPT-2 tokenizer
nlp = spacy.load("en_core_web_sm")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

text = "Beyond the above, emergent risks also include phenomena that might only exist in future language models or that have not yet been characterized in current language models."

def create_spacy_doc_from_gpt2(text):
    # Pre-tokenize the text using GPT-2
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    words = ['']
    spaces = []
    for i, tok in enumerate(gpt2_tokens):
        if not tok.startswith('Ġ') and tok.strip() not in string.punctuation:
            words[-1] += tok
            spaces.append(False)
        else:
            words.append(tok.strip('Ġ'))
            spaces.append(True)
    spaces = spaces[1:] + [False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    # Run the SpaCy pipeline on the custom Doc
    doc = nlp.get_pipe("tok2vec")(doc)
    doc = nlp.get_pipe("parser")(doc)
    return doc

doc = create_spacy_doc_from_gpt2(text)

# Print tokens and their dependencies
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")

"""FROM HERE"""


def calculate_offsets_with_spaces(gpt2_tokens, tokenizer):
    offsets = []
    idx = 0

    for token in gpt2_tokens:
        # Get the token text (with possible leading spaces)
        token_text = tokenizer.convert_tokens_to_string([token])

        # Strip leading spaces for comparison, but account for their length in offsets
        stripped_token_text = token_text.lstrip(' ').lstrip('Ġ')
        leading_space_len = len(token_text) - len(stripped_token_text)

        # Find the start index in the text
        start_idx = idx + leading_space_len
        end_idx = start_idx + len(stripped_token_text)

        # Append the offsets
        offsets.append((start_idx, end_idx))

        # Update the index for the next token
        idx = end_idx

    return offsets


def align_tokenizers(text, tokenizer, nlp):
    # Tokenize with GPT-2 tokenizer
    gpt2_tokens = tokenizer.tokenize(text)

    # Calculate offsets using token information
    gpt2_offsets = calculate_offsets_with_spaces(gpt2_tokens, tokenizer)

    # Process text with SpaCy
    doc = nlp(text)

    # Use SpaCy's Retokenizer to merge tokens
    with doc.retokenize() as retokenizer:
        for start, end in gpt2_offsets:
            span = doc.char_span(start, end)
            if span:
                retokenizer.merge(span)
            else:
                print(f"Warning: Span ({start}, {end}) does not align with SpaCy tokens.")

    return doc


# Example text
# text = """'I can't do it,' 'daigakunyuushishiken is too hard!' Ryunosuke said."""
# text = """20 or 30 grams per day), That means protein is"""
# text = tokenizer.decode([5381, 305, 431, 1022, 23077, 17570, 414, 290, 11982, 50152, 319, 13460, 326, 1057, 262, 9106, 315, 13, 632, 3636, 470, 307, 262, 717, 9048, 5337, 284, 1592, 262, 44127, 11, 475, 314, 4240, 611, 262, 9002, 290, 3096, 481, 467, 329, 340, 13, 5338, 4206, 30, 632, 1244, 880, 1592, 262, 11596, 13, 198, 464, 13146, 283, 286, 5896, 290, 5429, 78, 2523, 326, 317, 4757, 28828, 286, 28476, 34828, 3674, 64, 373, 645, 781, 4649, 13, 1770, 13, 1526, 430, 338, 3809, 318, 1241, 290, 7310, 11, 290, 287, 262, 31792, 21654, 286, 663, 27149, 7363, 881, 13357, 13, 314, 3636, 470, 307, 6655, 284, 766, 340, 3706, 355, 257, 8464, 393, 257, 2457, 396, 13, 198, 37, 689, 290, 376, 4740, 468, 617, 14081, 27149, 11, 475, 355, 1854, 423, 6515, 11, 262, 1621, 2346, 318, 257, 1643, 1035, 934, 13, 632, 857, 423, 883, 7188, 287, 543, 3307, 389, 5545, 287, 257, 8871, 5642, 11, 543, 318, 1223, 340, 468, 287, 2219, 351, 257, 1271, 286, 44127, 14591, 13, 6997, 13, 10299, 487, 318, 257, 845, 922, 6260, 11, 290, 996, 673, 338, 587, 19332, 8057, 329, 428, 5337, 11, 673, 468, 1865, 284, 1592, 257, 11596, 393, 5764, 13, 1406, 340, 1244, 880, 307, 1194, 2457, 396, 393, 8464, 13, 198, 1870, 644, 286, 383, 1632, 3149, 776, 7509, 30, 8730, 17924, 71, 42379, 6797, 351, 257, 7786, 4151, 290, 1027, 329, 1692, 9791, 4978, 287, 11359, 5917, 13, 679, 468, 587, 3688, 284, 11520, 28132, 11, 4502, 32226, 11, 290, 33089, 5030, 1377, 290, 883, 1115, 3588, 470, 1165, 427, 42457, 357, 3099, 387, 737, 632, 3636, 470, 5975, 502, 611, 428, 5337, 373, 257, 2457, 396, 11, 3863, 772, 257, 8464, 13, 198, 1870, 788, 11, 355, 3360, 4325, 11, 356, 1244, 651, 8754, 257, 12133, 1894, 416, 262, 44127, 3096, 290, 9002, 13, 1119, 1053, 1760, 340, 878, 13, 5338, 338, 284, 910, 340, 1839, 470, 1645, 757, 30, 198, 26392, 11, 314, 423, 645, 900, 2126, 319, 644, 481, 1592, 1282, 3321, 13, 1081, 1464, 11, 314, 1101, 4609, 284, 1064, 503, 13, 198, 40, 1549, 307, 845, 4609, 287, 534, 20681, 12, 12463, 10763, 0, 198, 31, 44, 301, 1069, 18558, 314, 1183, 766, 611, 314, 460, 651, 257, 4866, 422, 262, 5888, 0, 14373, 314, 423, 257, 1178, 584, 3835, 8358, 1739, 510, 284, 1100, 11, 475, 314, 1183, 1577, 340, 257, 2823, 0, 198, 10449, 345, 329, 262, 2882, 13, 314, 561, 1842, 284, 423, 12737, 284, 428, 1492, 422, 584, 319, 428, 2524, 284, 766, 703, 616, 9317, 4197, 351, 517, 5887, 9317, 13, 39947, 42, 314, 14765, 284, 2198, 503, 867, 286, 262, 3835, 345, 5610, 355, 880, 13, 198, 31, 44, 301, 1069, 18558, 314, 4398, 470, 1100, 428, 1492, 11, 475, 262, 3002, 3073, 5385, 13, 198, 31, 73, 69, 798, 82, 17, 314, 4236, 351, 523, 881, 286, 644, 345, 3417, 13, 314, 716, 991, 10427, 329, 317, 7703, 5155, 475, 314, 716, 2063, 835, 832, 13146, 283, 290, 611, 340, 7864, 11, 314, 481, 307, 655, 355, 3772, 13, 31342, 314, 423, 645, 2126, 703, 39528, 23201, 925, 262, 1351, 13, 843, 314, 1101, 991, 11263, 1521, 612, 318, 1239, 597, 3068, 286, 317, 797, 24592, 286, 38389, 13, 7875, 2687, 1100, 428, 1492, 30, 33058, 30, 198, 40, 366, 268, 2633, 276, 1, 317, 406, 22470, 2538, 36821, 11, 290, 314, 892, 340, 338, 13460, 389, 9090, 453, 1593, 11, 475, 287, 262, 886, 314, 2936, 340, 373, 8131, 19556, 13, 314, 2936, 6454, 12470, 546, 3336, 309, 4261, 21479, 46586, 1377, 257, 845, 922, 290, 366, 18049, 1, 1621, 11, 475, 351, 617, 4847, 326, 655, 1422, 470, 670, 329, 502, 13, 198, 14311, 584, 29467, 11, 314, 892, 314, 8359, 376, 29462, 5357, 376, 4261, 11015, 517, 621, 11096, 11, 475, 287, 616, 4459, 11, 340, 318, 635, 2192, 517, 19556, 13, 7735, 11, 340, 750, 407, 1730, 351, 262, 976, 1611, 286, 30573, 14, 82, 1733, 453, 366, 18049, 1, 357, 40, 716, 1262, 257, 3154, 14093, 994, 11, 20927, 502, 8, 13460, 326, 314, 561, 1842, 257, 44127, 8464, 284, 7239, 11, 2592, 618, 612, 389, 3689, 326, 466, 284, 3853, 422, 13, 7473, 51, 41884, 9447, 4146, 1546, 373, 257, 366, 11274, 1, 357, 4360, 691, 922, 8, 4947, 11, 475, 1839, 470, 1592, 5030, 465, 1218, 44127, 13, 314, 750, 407, 1100, 28163, 2149, 40, 20958, 11895, 3727, 50, 13, 198, 29435, 12346, 25, 3336, 19704, 7378, 12599, 14887, 1137, 532, 314, 836, 470, 892, 867, 661, 389, 3375, 546, 428, 530, 13, 632, 1595, 470, 3446, 423, 1605, 13460, 11, 355, 340, 318, 46190, 416, 257, 23618, 13997, 11, 475, 339, 318, 2877, 287, 262, 1294, 329, 257, 1588, 16058, 286, 262, 1492, 13, 10968, 11, 340, 318, 257, 1492, 326, 37928, 24234, 27962, 13, 198, 464, 44127, 15895, 468, 587, 4457, 26815, 287, 262, 938, 1178, 812, 11, 523, 314, 655, 1254, 317, 7703, 5155, 481, 1011, 340, 287, 262, 886, 351, 317, 17969, 329, 5985, 278, 6926, 290, 412, 42236, 355, 43736, 13, 6674, 772, 612, 714, 307, 257, 2368, 2457, 396, 290, 317, 7703, 5155, 2753, 340, 355, 262, 3015, 286, 262, 5583, 11, 407, 262, 9002, 13, 198, 9915, 5866, 357, 1212, 373, 257, 4004, 286, 6164, 329, 749, 286, 1853, 475, 340, 1422, 470, 1283, 284, 651, 326, 881, 9465, 11, 1049, 1492, 996, 8, 198, 40, 760, 383, 25688, 448, 318, 262, 3756, 24190, 11, 475, 314, 423, 407, 1100, 340, 1865, 355, 314, 4632, 655, 2822, 717, 3601, 654, 290, 428, 318, 17742, 1327, 284, 651, 11, 2158, 314, 423, 1100, 309, 1648, 416, 12568, 774, 11, 290, 326, 373, 407, 329, 502, 13, 198, 8421, 314, 467, 314, 561, 588, 284, 5875, 477, 286, 345, 329, 1194, 1049, 614, 286, 44127, 9984, 11, 340, 7061, 82, 257, 9476, 284, 2648, 534, 5608, 281, 13572, 11, 5875, 345, 13, 198, 40028, 2073, 326, 3947, 257, 1263, 1730, 618, 340, 1625, 503, 475, 543, 468, 1201, 587, 21655, 393, 24887, 30, 198, 7, 40, 1265, 11, 780, 612, 318, 636, 286, 502, 326, 5300, 345, 714, 766, 281, 10059, 11, 475, 34685, 11, 12750, 428, 614, 13, 1114, 1672, 11, 314, 714, 766, 366, 464, 36989, 1600, 366, 32, 7157, 13618, 5588, 284, 10005, 2185, 1600, 290, 366, 32, 17969, 329, 5985, 278, 6926, 1, 852, 262, 43736, 13, 33058, 10091, 198])
# nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer"])
# test = nlp(text)
# test.char_span(0,5)
# for sent in test.sents:
#     for word in sent:
#         print(word)
# doc = align_tokenizers(text, tokenizer, nlp)
# test.char_span(4077, 4079)
# text[4076:4078]
# for sent in doc.sents:
#     for word in sent:
#         print(word)
# # Check the resulting tokens
# print([token.text for token in doc])


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
nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer"])

ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# tokenizer.vocab['Ġcant']

auxs = [
    'Is', 'Are', 'Was', 'Were', 'Do', 'Does', 'Did', 'Wo', 'Would',
    'Have', 'Has', 'Had', 'Ca', 'Could', 'Might', 'Should', 'Ai']
special_cases = [[aux+"n't", [aux+'n', "'t"]] for aux in auxs]+\
    [[aux.lower()+"n't", [aux.lower()+'n', "'t"]] for aux in auxs]+\
    [[aux.upper()+"N'T", [aux.upper()+'N', "'T"]] for aux in auxs]+\
    [[aux+"n’t", [aux+'n', "’t"]] for aux in auxs]+\
    [[aux.lower()+"n’t", [aux.lower()+'n', "’t"]] for aux in auxs]+\
    [[aux.upper()+"N’T", [aux.upper()+'N', "’T"]] for aux in auxs]

for orth, toks in special_cases:
    nlp.tokenizer.add_special_case(orth, [{ORTH: tok} for tok in toks])
nlp.tokenizer.add_special_case("''s", [{ORTH: "''"}, {ORTH: "s"}])

pad = True
special_cases = []
# special_cases.extend([item.strip('Ġ') for item in tokenizer.vocab.keys()])
special_cases = [item.strip('Ġ') for item in tokenizer.vocab.keys() if len(item.strip('Ġ')) > 1]

for case in special_cases:
    nlp.tokenizer.add_special_case(case, [{ORTH: case}])

target_num_toks = 1_000_000_000  # should add up to 1b tokens
out_fp = os.path.join(DATA_DIR, 'cc100_1b_tokens_heads.jsonl')
words, words_used, wids, heads, toks, curr_toks, tok2word = [], [], [], [], [], [], []
ctx_size = 1024
keep_one = False

with open(out_fp, 'w') as f:
    # get data from HF
    data = load_dataset('cc100', 'en', streaming=True)
    data = data['train']
    pbar = tqdm.tqdm(total=target_num_toks)
    num_toks = 0

    # go over each text chunk: text chunk can be a whole artcile or just a paragraph, depending on the corpus
    for sample in data:
        # text = preprocess(sample)
        text = sample['text']
        text = re.sub(r",'", ", '", text)
        text = re.sub(r',"', ', "', text)
        text = re.sub(r"',", "' ,", text)
        text = re.sub(r'",', '" ,', text)
        text = re.sub(r"''", '"', text)
        text = re.sub(r"\.,", '. ,', text)
        text = re.sub(r"\)\.", ') .', text)
        if not text.strip('\n').strip():  # CC100 marks text boundary by \n\n
            toks.append(tokenizer.eos_token_id)
            words.append(tokenizer.eos_token)
            wids.append(-1)  # to be removed at training time
            heads.append(-1)  # to be removed at training time
            continue

        nlped = align_tokenizers(text, tokenizer, nlp)
        tokenized = tokenizer(text, truncation=False)
        toks.extend(tokenized['input_ids'])
        offset = -1
        for sent in nlped.sents:
            for word in sent:
                words.append(word.text)
                wids.append(word.i-offset-1)  # 0 indexed at each sentence
                heads.append(word.head.i-offset-1)  # 0 indexed at each sentence
                # rels.append(word.dep_)
            offset += len(sent)

        # if len(toks) < ctx_size:
        #     continue
        while len(toks) >= ctx_size:
            word_id = 0
            for i, tok in enumerate(toks[:ctx_size]):
                curr_toks.append(tok)
                print(tokenizer.decode(curr_toks).strip(),
                      '|||',
                      words[word_id].strip(),
                      '|||',
                      len(words[word_id]),
                      '|||',
                      num_toks+i,
                      # '|||',
                      # toks
                      )
                if tokenizer.decode(curr_toks).strip() == words[word_id].strip():
                    tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                    for tok_id in tok_ids:
                        if tok_id < 0:  # boundary situation can have negative tok_id
                            continue
                        tok2word.append(word_id)
                    word_id += 1
                    curr_toks = []
                if len(curr_toks) > 100:  # in the pretokenization script, token of length > 150 is removed
                    print(tokenizer.decode(curr_toks).strip(), words[word_id])
                    raise IOError

            if curr_toks:  # if the last token in batch NOT at the word boundary; e.g. [..., ddmin, nea][polis, ...]
                # print(f'LEFTOVER CURRTOKS: {tokenizer.decode(curr_toks)}')
                tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                for tok_id in tok_ids:
                    if tok_id < 0:  # boundary situation can have negative tok_id
                        continue
                    tok2word.append(word_id)
                keep_one = True
                # NO curr_toks = []
                # NO word_id += 1
            else:
                keep_one = False

            # get this batche's head and word boundary info
            # batch_wids = list(range(1, word_id+1))
            batch_wids = list(range(word_id))
            offsets = [batch_level - orig for orig, batch_level in zip(wids[:word_id], batch_wids)]
            batch_heads = [head + offset for head, offset in zip(heads[:word_id], offsets)]
            # print(len(tok2word))
            # print(tok2word[:100])
            assert len(tok2word) == ctx_size
            # print(len(batch_wids), len(batch_heads), word_id)
            assert len(batch_wids) == len(batch_heads) == word_id

            if pad:
                batch_heads = batch_heads + [-1]*(ctx_size-len(batch_heads))
            batch = {'input_ids': toks[:ctx_size],
                     'tok2word': tok2word,
                     'heads': batch_heads}
            f.write(json.dumps(batch)+'\n')

            # if keep_one:
            #     words, wids, heads = words[word_id-1:], wids[word_id-1:], heads[word_id-1:]
            # else:
            #     words, wids, heads = words[word_id:], wids[word_id:], heads[word_id:]
            words, wids, heads = words[word_id:], wids[word_id:], heads[word_id:]

            toks = toks[ctx_size:]
            tok2word = []

            num_toks += ctx_size
            pbar.update(ctx_size)

            if num_toks >= target_num_toks:
                print('done')
                break