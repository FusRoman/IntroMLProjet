import dataAnalysis as da
import numpy as np
import math
from collections import Counter
from itertools import product
import hashlib
import copy

# Tableau de feature dont les arguments sont :
# s => the sentence
# i => the index of the word
# t => the transform to apply (hash for exemple)
def features():
    return {
        'word': lambda s, i, t: t(s[i]),

        'pos': lambda s, i, t: i-2,
        'first': lambda s, i, t: i == 2,
        'last': lambda s, i, t: i == len(s) - 3,
        'len': lambda s, i, t: len(s[i]),

        'startUpper': lambda s, i, t: s[i][0].isupper(),
        'hasUpper': lambda s, i, t: any(c.isupper() for c in s[i]),
        'allUpper': lambda s, i, t: s[i].isupper(),
        #'all-alpha': lambda s, i, t: s[i].isalpha(),
        # 'has-digit':        lambda s, i, t: any(c.isdigit() for c in s[i]),
        'all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i]),

        'w-1': lambda s, i, t: t(s[i-1]),
        'w-2': lambda s, i, t: t(s[i-2]),
        'w+1': lambda s, i, t: t(s[i+1]),
        'w+2': lambda s, i, t: t(s[i+2]),

        #'w-1-all-alpha': lambda s, i, t: s[i-1].isalpha(),
        #'w-1-all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i-1]),
        #'w+1-all-alpha': lambda s, i, t: s[i+1].isalpha(),
        #'w+1-all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i+1]),

        'prefix-1': lambda s, i, t: t(s[i][:1]),
        'prefix-2': lambda s, i, t: t(s[i][:2]),
        'prefix-3': lambda s, i, t: t(s[i][:3]),

        'suffix-1': lambda s, i, t: t(s[i][-1:]),
        'suffix-2': lambda s, i, t: t(s[i][-2:]),
        'suffix-3': lambda s, i, t: t(s[i][-3:]),
        
        #'voyel-ratio':   lambda s, i, t: (lambda w:sum(w.count(v) for v in "aeiouy") / len(w) )(s[i].lower())
    }

# permet d'extraire les features d'une phrases


def extract_features(s, i, transform=lambda x: x):
    return tuple(f(s, i, transform) for f in features().values())


def preprocessSentences(dataset):
    begin_s = "<s>"
    end_s = "</s>"
    for d in dataset.data:
        d.sentence = [begin_s, begin_s] + d.sentence + [end_s, end_s]
        d.labels = [begin_s, begin_s] + d.labels + [end_s, end_s]

# extrait les features de tous le corpus train

def buildFeature(dataset):
    preprocessSentences(dataset)
    X = []
    Y = []
    for d in dataset.data:
        X += [extract_features(d.sentence, i, transform=hash)
              for i in range(2, len(d.sentence) - 2)]
        Y += d.labels[2:-2]
    return X, Y
        