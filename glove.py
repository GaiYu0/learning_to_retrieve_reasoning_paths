import pickle
import sys

import numpy as np

prefix = sys.argv[1]
words = [line.split(' ') for line in open(f'{prefix}.txt')]
word2index = dict(zip([x[0] for x in words], range(len(words))))
embeddings = np.array([x[1:] for x in words], dtype=np.float)
_embeddings = np.vstack([embeddings, embeddings.mean(0)])
pickle.dump(word2index, open(f'{prefix}.word2index', 'wb'))
np.save(f'{prefix}.embeddings', embeddings)
