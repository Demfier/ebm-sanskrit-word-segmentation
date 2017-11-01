import pickle
import numpy as np
import operator
import math
import time
from sklearn.feature_selection import mutual_info_regression
from functools import partial

from multiprocessing import Pool
pool = Pool(processes=20)


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


def correlation_task(meta_path, pmi_vector):
    return(tuple((pearson_def(meta_path, pmi_vector))))


def correlation_task_for8l(df):
    pmi = []
    for each in range(9960):
        pmi.append(df['targetDict'][each][3])
    corr = []
    start = time.time()
    for each in range(788992):
        corr.append(tuple((each, pearson_def(df['featureMatrix'][each], pmi))))
    print('Took %f seconds' % (time.time() - start))
    corr.sort(reverse=True, key=operator.itemgetter(1))
    return(corr)


mat_load_start = time.time()
df = pickle.load(open("10K_1L.p", "rb"))
print('1L loading took %f seconds' % (time.time() - mat_load_start))
pmi_vector = pickle.load(open('all_pmi.p', 'rb'))
pmi_func = partial(correlation_task, list(pmi_vector.values()))

fs_corr = time.time()
tenk_features = pool.map(pmi_func, df['featureMatrix'])
print('Took %f seconds for 1L to 10K corr' % (time.time() - fs_corr))
tenk_features.sort(reverse=True, key=operator.itemgetter(0))
corr = corr[:10000]
pickle.dump(corr, open("corr4.p", "wb"))
