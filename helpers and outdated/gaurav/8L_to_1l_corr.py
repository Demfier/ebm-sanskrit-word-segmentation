import pickle
import numpy as np
import time
import operator
from functools import partial
import pandas as pd
import math


def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)


def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


def correlation_task_for8l(feature_vector, pmi):
    return(tuple((each, pearson_def(feature_vector, pmi))))

file_prefix = 'featureSet_500samples_8L_'
files_list = [(file_prefix + str(i) + '.p') for i in range(20)]

featureMatrix = np.array([])
for idx, featureFile in enumerate(files_list):
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    if idx == 0:
        featureMatrix = df['featureMatrix']
        continue
    featureMatrix = np.concatenate((featureMatrix, df['featureMatrix']), axis=1)

pmi = pickle.load(open('all_pmi.p', 'rb'))
pmi_func = partial(correlation_task_for8l, list(pmi.values()))
fs_start_corr = time.time()
one_l_features = pool.map(pmi_func, featureMatrix)
print('Took %f seconds for 8L to 1L corr' % (time.time() - fs_start_corr))
one_l_features = one_l_features.sort(reverse=True, key=operator.itemgetter(1))
