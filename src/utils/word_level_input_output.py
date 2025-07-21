"""
Author: Tatsuya
"""
import os
import torch, re, sys
from tqdm import tqdm

class WordLevelIO:

    def __init__(self, data_fp, tokenizer, ctx_size, stride, batch_size, device, scale=True):
        # print('before init of WLIO')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        self.ctx_size = ctx_size
        self.stride = stride
        self.batch_size = batch_size
        self.device = device
        self.scale = scale
        self.words, self.wids, self.gid2info, self.wid2gid =\
            self._get_data(data_fp)
        self.batches, self.tok2word, self.word2tok, self.toks =\
            self._tokenize_and_batchify(tokenizer)
        self.batched_attentions = None
        # print('after init of WLIO')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    def _get_data(self, data_fp):
        data = [line.strip().split('\t') for line in open(data_fp).readlines() if line.strip()]
        wid2gid = {line[0]:i for i, line in enumerate(data)}
        gid2info = {i: line for i, line in enumerate(data)}
        return [line[1] for line in data], [line[0] for line in data], gid2info, wid2gid


    def _tokenize_and_batchify(self, tokenizer):
        # docs, codes = get_data(os.path.join(DATA_DIR, 'ewt.txt'))
        concat = ' '.join(self.words)
        concat = re.sub(r" <\|endoftext\|> ", r"<|endoftext|>", concat)
        toks = tokenizer(concat)
        # create token to word mapping
        curr_toks = []
        word_id = 0
        tok2word = dict()
        word2tok = dict()
        for i, tok in enumerate(toks['input_ids']):
            curr_toks.append(tok)
            if tokenizer.decode(curr_toks).strip() == self.words[word_id]:
                tok_ids = list(range(i-len(curr_toks)+1, i+1))
                for tok_id in tok_ids:
                    tok2word[tok_id] = word_id
                word2tok[word_id] = tok_ids
                assert tokenizer.decode(toks['input_ids'][i-len(curr_toks)+1:i+1]).strip() == self.words[word_id]
                # print(tokenizer.decode(toks['input_ids'][i-len(curr_toks)+1:i+1]).strip(), words[word_id])
                word_id += 1
                curr_toks = []
        input_ids = toks['input_ids']
        batch = []
        batches = []
        num_toks = 0
        for i in range(0, len(input_ids), self.stride):
            batch.append(input_ids[i:i+self.ctx_size])
            if len(batch[-1]) < self.ctx_size:  # discard the last small batch
                break
            if len(batch) == self.batch_size:
                batches.append(torch.tensor(batch))
                num_toks += self.ctx_size + self.stride * (self.batch_size-1)
                batch = []
        # batches.append(torch.tensor(batch))
        tok2word = {tid:tok2word[tid] for tid in range(num_toks)}
        num_words = tok2word[num_toks-1] + 1
        word2tok = {wid:word2tok[wid] for wid in word2tok if wid < num_words}
        return batches, tok2word, word2tok, toks

    def _get_batched_attentions(self, model):
        model.eval()
        model.to(self.device)

        # batches, tok2word, toks = _tokenize_and_batchify(tokenizer, ctx_size, stride, batch_size, words)
        attentions = []
        with torch.no_grad():
            for batch in tqdm(self.batches):
                # output = model(batch.to(device), output_attentions=True)
                output = model(batch.to(self.device), output_attentions=True)
                # original: (layer, batch_size, head, source, target)
                # reshaped: (batch_size, layer, head, source, target)
                attention = torch.stack(output.attentions, dim=1)
                attentions.append(attention)
                del output
                # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        return attentions


    # original: (num_batches, layer, batch_size, head, source, target)
    # reshaped: (num_sequences=num_batch*batch_size, layer, head, source, target)

    def _to_word_level_attentions(self, attentions):
        """

        :param attentions: [torch(shape: (layer, head, source, target)), ...]
        :return:
        """
        seqs = torch.cat(self.batches, dim=0)
        attentions = torch.cat(attentions, dim=0)
        word_level_attentions = []
        for seq_id, seq in enumerate(attentions):
            # seq has size: (layer, head, source token, target token)

            cml = int(self.stride * seq_id)
            assert seqs[seq_id].tolist() == self.toks['input_ids'][cml:cml + self.ctx_size]

            ### all "attended to" tokens should be *summed* over constituent tokens to map to word level
            # word id each tok corresponds to e.g. [0, 0, 1, 1, 1, 2, 3, ...] (first 2 toks correspond to word 0)
            word_membership = torch.tensor([self.tok2word[i]-self.tok2word[cml] for i in range(cml, cml + self.ctx_size)],
                                           dtype=torch.int64, device=self.device)
            # number of words (<=ctx_size)
            num_words = len(set(word_membership.tolist()))
            # reshape the membership tensor to match the source tensor (original token level attention tensor)
            word_membership = word_membership.expand(*seq.shape[:-1], -1)
            # initialize the word level tensor to all zeros
            target_summed = torch.zeros(*seq.shape[:-1], num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            target_summed.scatter_add_(dim=-1, index=word_membership, src=seq)  # (layer, head, source_token, target_word)

            ### all "attended from" tokens should be *averaged* over constituent tokens to map to word level
            word_membership = torch.tensor([self.tok2word[i]-self.tok2word[cml] for i in range(cml, cml + self.ctx_size)],
                                           dtype=torch.int64, device=self.device)
            target_summed = target_summed.permute(0, 1, 3, 2)  # (layer, head, target=word, source=tok)
            word_membership = word_membership.expand(*target_summed.shape[:-1], -1)  # layer, head, word, tok

            # initialize the word level tensor to all zeros
            source_averaged = torch.zeros(*target_summed.shape[:-2], num_words, num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            source_averaged.scatter_add_(dim=-1, index=word_membership,
                                         src=target_summed)  # (layer, head, source_token, target_word)
            word_membership_counts = torch.zeros_like(source_averaged,
                                                      device=self.device)  # initialize constituent token count vector
            word_membership_counts.scatter_add_(
                dim=-1,
                index=word_membership,
                src=torch.ones_like(target_summed)
            )
            source_averaged = source_averaged / word_membership_counts
            source_averaged = source_averaged.permute(0, 1, 3, 2)  # back to the original
            source_averaged.to('cpu')  # offload it to CPU
            word_level_attentions.append(source_averaged)

        return word_level_attentions

    def get_attentions(self, model, unit='word', output=False):
        batched_attentions = self._get_batched_attentions(model)
        if unit == 'word':
            batched_attentions = self._to_word_level_attentions(batched_attentions)
        self.batched_attentions = batched_attentions
        if output:
            return batched_attentions

### BETA FEATURES

    def _to_word_level_attentions_single_batch(self, cml_num_seqs, attentions):
        """

        :param attentions: [torch(shape: (layer, head, source, target)), ...]
        :return:
        """
        seqs = torch.cat(self.batches, dim=0)
        # attentions = torch.cat(attentions, dim=0)
        word_level_attentions = []
        for seq_id, seq in enumerate(attentions):
            # seq has size: (layer, head, source token, target token)

            cml = int(self.stride * (cml_num_seqs+seq_id))
            assert seqs[cml_num_seqs+seq_id].tolist() == self.toks['input_ids'][cml:cml + self.ctx_size]

            ### all "attended to" tokens should be *summed* over constituent tokens to map to word level
            # word id each tok corresponds to e.g. [0, 0, 1, 1, 1, 2, 3, ...] (first 2 toks correspond to word 0)
            word_membership = torch.tensor([self.tok2word[i]-self.tok2word[cml] for i in range(cml, cml + self.ctx_size)],
                                           dtype=torch.int64, device=self.device)
            # number of words (<=ctx_size)
            num_words = len(set(word_membership.tolist()))
            # reshape the membership tensor to match the source tensor (original token level attention tensor)
            word_membership = word_membership.expand(*seq.shape[:-1], -1)
            # initialize the word level tensor to all zeros
            target_summed = torch.zeros(*seq.shape[:-1], num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            target_summed.scatter_add_(dim=-1, index=word_membership, src=seq)  # (layer, head, source_token, target_word)

            ### all "attended from" tokens should be *averaged* over constituent tokens to map to word level
            word_membership = torch.tensor([self.tok2word[i]-self.tok2word[cml] for i in range(cml, cml + self.ctx_size)],
                                           dtype=torch.int64, device=self.device)
            target_summed = target_summed.permute(0, 1, 3, 2)  # (layer, head, target=word, source=tok)
            word_membership = word_membership.expand(*target_summed.shape[:-1], -1)  # layer, head, word, tok

            # initialize the word level tensor to all zeros
            source_averaged = torch.zeros(*target_summed.shape[:-2], num_words, num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            source_averaged.scatter_add_(dim=-1, index=word_membership,
                                         src=target_summed)  # (layer, head, source_token, target_word)
            word_membership_counts = torch.zeros_like(source_averaged,
                                                      device=self.device)  # initialize constituent token count vector
            word_membership_counts.scatter_add_(
                dim=-1,
                index=word_membership,
                src=torch.ones_like(target_summed)
            )
            source_averaged = source_averaged / word_membership_counts
            source_averaged = source_averaged.permute(0, 1, 3, 2)  # back to the original
            source_averaged.to('cpu')  # offload it to CPU
            word_level_attentions.append(source_averaged)

        return word_level_attentions
    def _generate_batched_attentions(self, model):
        model.eval()
        model.to(self.device)

        # batches, tok2word, toks = _tokenize_and_batchify(tokenizer, ctx_size, stride, batch_size, words)
        with torch.no_grad():
            for batch in tqdm(self.batches):
                # output = model(batch.to(device), output_attentions=True)
                output = model(batch.to(self.device), output_attentions=True)
                # original: (layer, batch_size, head, source, target)
                # reshaped: (batch_size, layer, head, source, target)
                attention = torch.stack(output.attentions, dim=1)
                del output
                yield attention

    def _head_probe(self, batch, num_seqs, num_words):
        """
        x_i -> x_j attention and x_i <- x_j attention
        e.g.
        1       0       0       0       0
        0.3     0.7     0       0       0
        0.2     0.2     0.6     0       0
        0.1     0.1     0.5     0.4     0
        0.01    0.09    0.3     0.3     0.3
        -> probe says: maxes that go through the diagonal points are the parents

        1. scaling based on the number of 'attendable' tokens
        2. crude 1024 with 1024 stride vs sentence level
        """
        assert self.ctx_size == self.stride, 'Currently this probe only supports non-overlapping sliding window.'
        first_word_overlaps = False  # if the first word of this batch (not seq) overlaps with the previous word
        by_head_predictions = []
        for seq_id, seq in enumerate(batch):
            seq.to(self.device)
            first_tok_id = self.ctx_size * (num_seqs+seq_id)
            # if final token in the last seq and the first token in the current batch belong to the same word
            if self.tok2word[first_tok_id] == num_words-1:
                if seq_id == 0:
                    first_word_overlaps = True
                by_head_predictions = by_head_predictions[:-1]  # just use the current token
                num_words -= 1  # consider this boundary 'taken care of'
            if self.scale:
                scaling_factor = torch.tensor(list(range(1, seq.shape[-1]+1)), device=self.device)
                scaling_factor = scaling_factor.expand(*seq.shape)
                seq = seq * scaling_factor  # seq should have a shape (layer, head, source, target)
            both = torch.cat([seq, seq.permute(0, 1, 3, 2)], dim=-1)
            max_ids = torch.argmax(both, dim=-1)
            # source-target concatenated, so fix the id, and change seq-specific id to global id
            max_ids = [idx%seq.shape[-1]+num_words for idx in max_ids.flatten()]
            max_ids = torch.tensor(max_ids, device=self.device).reshape(*seq.shape[:-1])  # put it back to the original shape
            max_ids = max_ids.permute(2,0,1).view(seq.shape[-1], seq.shape[0], seq.shape[1])  # (word, layer, head)
            by_head_predictions.extend(max_ids.tolist())
            num_words += seq.shape[-1]
        return by_head_predictions, first_word_overlaps

    def get_sas_preds(self, model, unit='word'):
        batched_attentions_generator = self._generate_batched_attentions(model)
        batch_counter = 0
        num_words = 0
        by_head_predictions = []
        for batched_attentions in batched_attentions_generator:
            torch.cuda.empty_cache()
            # def sizeof_fmt(num, suffix='B'):
            #     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
            #     for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            #         if abs(num) < 1024.0:
            #             return "%3.1f %s%s" % (num, unit, suffix)
            #         num /= 1024.0
            #     return "%.1f %s%s" % (num, 'Yi', suffix)
            #
            # for tensor in [batched_attentions]:
            #     if type(tensor) == list:
            #         memory_bytes = tensor[0].element_size() * tensor[0].nelement() * len(tensor)
            #         memory_mb = memory_bytes / (1024 ** 2)
            #     elif type(tensor) == torch.Tensor:
            #         memory_bytes = tensor.element_size() * tensor.nelement()
            #         memory_mb = memory_bytes / (1024 ** 2)
            # print(f"Tensor memory usage: {memory_mb:.2f} MiB")
            #
            # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #         locals().items())), key=lambda x: -x[1])[:10]:
            #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            if unit == 'word':
                batched_attentions = self._to_word_level_attentions_single_batch(
                    cml_num_seqs=batch_counter*self.batch_size,
                    attentions=batched_attentions
                )
            batch_by_head_predictions, first_word_overlaps = self._head_probe(
                batch=batched_attentions,
                num_words=num_words,
                num_seqs=batch_counter*self.batch_size
            )
            batch_counter += 1
            num_words += len(batch_by_head_predictions)
            if first_word_overlaps:
                num_words -= 1
                by_head_predictions.pop()  # pop the overlapping element
            by_head_predictions.extend(batch_by_head_predictions)
        print(len(by_head_predictions), len(self.word2tok))
        assert len(by_head_predictions) == len(self.word2tok), 'Numbers of words do not match.'
        return by_head_predictions

"""TEST

from transformers import AutoTokenizer
from ablated_pythia import AblationGPTNeoXForCausalLM

data_fp = os.path.join(os.getcwd(), 'data', 'ewt.txt')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
model = AblationGPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
ctx_size = 1024
stride = 512
batch_size = 2
device = 'cpu'
wlio = WordLevelIO(data_fp, tokenizer, ctx_size, stride, batch_size, device)
attentions = wlio.get_attentions(model, 'word')
for seq in attentions:
    print(seq.shape)
"""