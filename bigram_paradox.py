import random

from nltk.corpus import brown

# pos tags
tags = brown.tagged_words()
tags[:10]
target = 'AT'
# unigram
def random_unigrams_det_count():
    idxs = random.sample(range(len(tags)), k=2)
    if tags[idxs[0]][1] == target and tags[idxs[1]][1] == target:
        return 2
    elif tags[idxs[0]][1] == target or tags[idxs[1]][1] == target:
        return 1
    else:
        return 0

# bigram
def random_bigram_det_count():
    idx = random.randint(0, len(tags)-2)
    if tags[idx][1] == target:
        if tags[idx-1][1] == target:
            return 2
        else:
            return 1
    elif tags[idx-1][1] == target:
        if tags[idx][1] == target:
            return 2
        else:
            return 1
    else:
        return 0

k = 10_000

su = []
for i in range(k):
    su.append(random_unigrams_det_count()) # unigram
print(sum(su)/k)

sb = []
for i in range(k):
    sb.append(random_bigram_det_count()) # bigram
print(sum(sb)/k)
