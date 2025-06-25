"""
Integrated a separate conllu2json.py
-> no ' '.join(words), so the text is faithful to the original corpus (no extra whitespaces)
"""
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os
import pathlib, tqdm, spacy, re, json#, torch
import string
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from spacy.tokens import Doc

# tokenizer('£')
# tokenizer.decode([14988])
# len(outputs['input_ids'])
# toks = tokenizer.tokenize('Sotheby’s International Realty Canada.')
# len(toks)
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# test = tokenizer.tokenize('SothebyâĢĻs')
# test = tokenizer.tokenize('Sotheby’s')
# for tok in test:
#     print(tok)
# [tokenizer.tokenize(tok) for tok in test]
# mappings = [
#     ['âĢĻ', r'’'],
#     ['Ċ', r'\n']
# ]
# re.sub('Ċ', '\n', '.Ċ')
def nlp_with_pretokenized_text(text, tokenizer, nlp):
    # Pre-tokenize the text using GPT-2
    tokens = tokenizer.tokenize(text)
    ids = tokenizer(text)['input_ids']
    words = []
    spaces = []
    word = []
    for i, tok in enumerate(tokens):
        # if the token is a continuation of the previous word and if it's not a punctuation
        if not tok.startswith('Ġ') and tok.strip() not in string.punctuation:
            # if this is a start of a new word, specify that there's no space
            if not word:
                spaces.append(False)
            word.append(ids[i])
        # if the token starts with a space (Ġ) it's a new word
        else:
            # remaining toks
            if word:
                decoded = tokenizer.decode(word).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                else:
                    spaces.pop()
                word = []
            if tok in string.punctuation:
                decoded = tokenizer.decode(ids[i]).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                    spaces.append(False)
            else:
                decoded = tokenizer.decode(ids[i]).strip().strip('Ġ')
                if decoded:
                    words.append(decoded)
                    spaces.append(True)
    # last remaining tokens
    if word:
        decoded = tokenizer.decode(word).strip().strip('Ġ')
        if decoded:
            words.append(decoded)
        else:
            spaces.pop()
    spaces = spaces[1:] + [False]
    if '' in words:
        print(words, tokens)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    # Run the SpaCy pipeline on the custom Doc
    doc = nlp.get_pipe("tok2vec")(doc)
    doc = nlp.get_pipe("parser")(doc)
    return doc

def preprocess(sample):
    # text = sample['text'].strip('\n').strip()
    text = sample['text'].strip()
    if not text or max([len(word) for word in text.split()]) > 150 or len(text) > 1_000_000:
        return None
    # text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f]', ' ', text)
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\xa0\t\n\r\f\v\x00-\x1F\x85]', ' ', text)
    text = re.sub(r' +', ' ', text)
    if not text:
        return None
    return text

ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer"])
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
pad = True

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
        text = preprocess(sample)
        if not text:
            continue
        # test = tokenizer.decode([
        #     262, 13325, 2370, 286, 7815, 2891, 779, 1683, 1043, 6609, 287, 262, 995, 3693, 16, 60, 1, 2514, 6901, 597, 2877,
        #     1448, 355, 705, 19795, 1800, 6, 393, 705, 34346, 7129, 6, 16857, 15565, 326, 484, 389, 2877, 10826, 286, 617, 2961,
        #     3800, 286, 1692, 2478, 326, 262, 3741, 286, 47634, 468, 1364, 2157, 13, 1114, 617, 11, 428, 714, 307, 257, 3967,
        #     6764, 11, 30253, 11, 329, 1672, 11, 326, 884, 2628, 2107, 287, 3744, 22471, 351, 3450, 19424, 1114, 1854, 11, 2644,
        #     705, 19795, 1800, 6, 318, 257, 4633, 2095, 5612, 13, 1114, 606, 11, 705, 19795, 1800, 6, 43397, 25086, 779, 286,
        #     4133, 290, 8889, 286, 262, 9028, 290, 6573, 5423, 286, 705, 37636, 1417, 6, 1692, 14515, 1106, 3574, 262, 22116,
        #     286, 17911, 2770, 3725, 11, 1111, 777, 5009, 389, 8603, 530, 12, 22339, 290, 35010, 526, 1, 1026, 373, 1903, 6939,
        #     326, 262, 1115, 11379, 7297, 286, 3968, 656, 8026, 11, 19461, 290, 7931, 21449, 8197, 287, 262, 29666, 4289, 329,
        #     2031, 550, 645, 19648, 287, 5478, 2354, 262, 38063, 19272, 526, 3886, 16171, 4381, 11, 38967, 2461, 262, 5370, 286,
        #     262, 5961, 12, 43032, 3162, 286, 3771, 23569, 11, 543, 11185, 790, 1440, 812, 284, 10568, 30971, 1597, 3181, 878,
        #     340, 13, 1024, 37061, 389, 1682, 3230, 26, 262, 4009, 2753, 663, 1438, 422, 262, 7243, 13, 5593, 1004, 539, 88,
        #     12007, 262, 717, 530, 287, 399, 18131, 8482, 287, 21709, 13, 632, 8197, 46653, 290, 34241, 338, 513, 12, 14247,
        #     1080, 379, 326, 640, 11, 262, 9539, 284, 307, 1444, 12556, 11, 6046, 290, 11450, 13, 61, 366, 33, 577, 11, 2807,
        #     1911, 1867, 8314, 632, 22728, 284, 307, 5524, 30, 13, 40131, 2351, 9594, 286, 12068, 7443, 13, 31439, 3640, 290,
        #     262, 287, 12, 18053, 3781, 286, 7228, 10691, 422, 262, 8026, 7129, 7603, 1728, 25797, 290, 9056, 286, 262, 661,
        #     287, 883, 44741, 1661, 13, 632, 318, 783, 4762, 326, 4568, 286, 262, 8026, 7129, 5384, 1816, 3675, 262, 7103, 5359,
        #     286, 13834, 870, 2057, 11, 1767, 3002, 654, 11, 290, 23629, 13, 17377, 41302, 11270, 284, 1918, 290, 23867, 547,
        #     19893, 11, 996, 3729, 28742, 287, 3918, 290, 9706, 1022, 13817, 13, 464, 3616, 286, 867, 286, 777, 21641, 3793,
        #     6439, 13, 1119, 743, 423, 587, 973, 329, 21819, 25797, 13, 383, 4695, 389, 11791, 416, 5895, 326, 1950, 257, 1744,
        #     5536, 779, 13, 19408, 12, 2339, 14354, 287, 406, 3372, 14644, 389, 3360, 16173, 355, 11845, 393, 435, 46870, 779,
        #     11, 475, 262, 2370, 3793, 6179, 876, 3693, 5237, 60, 6719, 31304, 1242, 318, 7424, 287, 262, 20316, 13, 3771, 31304,
        #     2647, 318, 41240, 422, 1043, 12834, 11, 981, 1582, 21587, 1242, 460, 307, 1043, 319, 12586, 286, 597, 1611, 13, 383,
        #     6846, 389, 4273, 3828, 306, 746, 82, 290, 3881, 21641, 13, 383, 1242, 743, 393, 743, 407, 423, 550, 257, 4158, 2163,
        #     13, 128, 254, 70, 415, 34655, 27081, 11, 1514, 10872, 13, 2773, 286, 262, 995, 338, 13325, 1479, 12, 5646, 8573, 13,
        #     464, 25733, 547, 7633, 1417, 2884, 262, 347, 1586, 1956, 7696, 543, 373, 7362, 1141, 428, 2278, 416, 2793, 5417,
        #     2974, 13, 2312, 661, 389, 1444, 262, 46840, 12, 5497, 1547, 11, 290, 262, 14555, 6292, 9667, 389, 883, 286, 262,
        #     1012, 709, 271, 3968, 5043, 11, 617, 1511, 11, 4059, 812, 2084, 13, 40713, 453, 11, 14515, 547, 19177, 12, 41268,
        #     1456, 3808, 475, 2370, 286, 7915, 18413, 6140, 284, 1656, 287, 262, 3094, 4996, 286, 7815, 2891, 3858, 852, 4166,
        #     284, 6050, 845, 1180, 12493, 13, 2953, 705, 52, 1350, 19830, 3972, 262, 8849, 319, 262, 11945, 286, 262, 5044, 4693,
        #     1043, 612, 7603, 326, 262, 11372, 286, 262, 4899, 475, 2395, 445, 262, 12847, 286, 1588, 23311, 11, 281, 3842, 326,
        #     468, 587, 28569, 366, 1416, 4005, 2667, 18161, 1821, 60, 1318, 389, 645, 2877, 18570, 11, 4249, 750, 484, 1429,
        #     11945, 284, 7330, 262, 44173, 13, 2312, 4568, 2314, 307, 7247, 4361, 355, 262, 691, 393, 772, 262, 7226, 3034, 3842,
        #     286, 367, 6351, 504, 13, 5334, 5353, 547, 21792, 25, 484, 547, 7525, 34102, 262, 6174, 286, 327, 712, 2340, 17414,
        #     3901, 60, 543, 318, 6108, 284, 423, 587, 1695, 1231, 15106, 4386, 329, 510, 284, 1440, 1528, 706, 262, 1494, 13,
        #     2215, 1844, 11, 262, 1115, 12, 8095, 88, 2615, 481, 28889, 286, 2310, 530, 12, 36269, 290, 1596, 734, 12, 36269,
        #     19592, 11, 543, 481, 4414, 422, 7850, 5009, 290, 2562, 1895, 284, 262, 3240, 7372, 13, 464, 19592, 481, 22265, 257,
        #     6994, 4067, 326, 468, 587, 23957, 329, 257, 1271, 286, 812, 290, 1650, 7848, 257, 9566, 3939, 3716, 11, 1363, 284,
        #     7117, 2097, 11, 257, 9208, 636, 286, 262, 3240, 447, 247, 82, 45293, 15012, 13, 818, 262, 7791, 447, 247, 82, 1486,
        #     11, 7695, 3019, 14651, 468, 587, 25923, 286, 262, 1989, 447, 247, 82, 2106, 290, 15012, 11, 475, 635, 20915, 262,
        #     761, 329, 257, 11811, 19713, 326, 561, 1037, 284, 3708, 262, 27597, 290, 32951, 5612, 286, 520, 3400, 9458, 13,
        #     20854, 768, 373, 6325, 287, 2805, 428, 614, 11, 351, 670, 319, 2524, 3726, 287, 2932, 13, 383, 2615, 318, 900, 329,
        #     11939, 287, 23608, 2864, 13, 36, 41807, 5108, 439, 388, 11, 22669, 379, 7695, 3019, 14651, 29778, 11, 531, 25, 564,
        #     250, 3260, 262, 1327, 670, 356, 1234, 656, 262, 1486, 286, 262, 7791, 11, 340, 447, 247, 82, 1049, 284, 766, 5103,
        #     2221, 319, 257, 2615, 543, 481, 787, 257, 1049, 10156, 284, 262, 27597, 286, 520, 3400, 9458, 13, 447, 250, 464,
        #     2615, 447, 247, 82, 1486, 32067, 7612, 286, 262, 3240, 447, 247, 82, 5527, 3939, 15012, 351, 257, 517, 11811, 3918,
        #     11, 588, 326, 286, 23939, 2097, 319, 262, 29272, 2524, 13, 383, 1255, 481, 307, 257, 3660, 290, 30511, 2478, 543,
        #     318, 8564, 284, 262, 1957, 21334, 290, 356, 804, 2651, 284, 4379, 340, 1844, 13, 447, 251, 1722, 262, 749, 2274,
        #     3090, 284, 262, 26048, 7779, 12750, 11, 262, 11164, 14208, 1838, 2407, 281, 5726, 13, 3242, 1586, 262, 749, 15892,
        #     2272, 287, 663, 1398, 11, 1863, 351, 257, 3210, 48019, 569, 23, 3113, 11, 3624, 12, 6603, 6540, 24800, 11, 290,
        #     1115, 12, 392, 12, 64, 12, 13959, 10860, 286, 284, 5469, 5339, 11, 262, 11164, 14208, 318, 530, 286, 262, 749,
        #     21362, 6332, 12, 315, 879, 5672, 319, 262, 1910, 13
        # ])

        # text = sample['text']
        # text = test
        # num = len(tokenizer.tokenize(text))
        # num = len(text.split())
        # num_toks += num
        # pbar.update(num)
        # if 'is a terrible system of economics' not in text:
        #     continue
        # print(text)
        # print(tokenizer(text))['input_ids']
        # break
        if not text.strip('\n').strip():  # CC100 marks text boundary by \n\n
            toks.append(tokenizer.eos_token_id)
            words.append(tokenizer.eos_token)
            wids.append(-1)  # to be removed at training time
            heads.append(-1)  # to be removed at training time
            continue

        nlped = nlp_with_pretokenized_text(text, tokenizer, nlp)
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
        while len(toks) > ctx_size:
            word_id = 0
            for i, tok in enumerate(toks[:ctx_size]):
                curr_toks.append(tok)
                # print(tokenizer.decode(curr_toks).strip(),
                #       '|||',
                #       words[word_id].strip(),
                #       '|||',
                #       len(words[word_id]),
                #       '|||',
                #       num_toks+i,
                #       '|||',
                #       toks
                #       )
                # try:
                #     a = words[word_id].strip()
                # except:
                #     print(curr_toks, tokenizer.decode(curr_toks), word_id, len(words), words, toks)
                if tokenizer.decode(curr_toks).strip('Ġ').strip() == words[word_id].strip('Ġ').strip():
                # if tokenizer.decode(curr_toks).strip() == words[word_id].strip():
                # if [tokenizer(tokenizer.decode(tok).strip())['input_ids'] for tok in curr_toks] == words[word_id].strip():
                    tok_ids = list(range(i - len(curr_toks) + 1, i + 1))
                    for tok_id in tok_ids:
                        if tok_id < 0:  # boundary situation can have negative tok_id
                            continue
                        tok2word.append(word_id)
                    word_id += 1
                    curr_toks = []
                if len(curr_toks) > 300:  # in the pretokenization script, token of length > 150 is removed
                    print('too many curr toks')
                    print(tokenizer.decode(curr_toks).strip(), words[word_id], toks)
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