import pickle
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#
print("Loading pickle file...")
# print("Starting reading features")
# for featVec in featureMatrix:


# from pandas import read_pickle
# load data
# names = range(500)
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

feature_set = []
count = 1
array = pickle.load(open('featureSet_500samples_8L_0_new.p', 'rb'), encoding='utf8')

for i, featVec1 in enumerate(array[:100]):
    for j, featVec2 in enumerate(array[i + 1: i + 100]):
        corr = pearson_def(featVec1, featVec2)
        if corr > 0.90:
            print(count, corr)
            count += 1
            print(type(featVec1), type(featVec2))
            feature_set.append(featVec1)
            feature_set.append(featVec2)

print(len(feature_set), count)
# feature_set = feature_set
with open('final_features_0.p', 'wb') as ff:
    pickle.dump(feature_set, ff)
# print("finished loading pickel file.")
# X = np.array(array).transpose()
# Y = np.array(array)
# # feature extraction
# model = LogisticRegression()
# rfe = RFE(model, 500)
# fit = rfe.fit(X, Y)
# print("Num Features: %d") % fit.n_features_
# print("Selected Features: %s") % fit.support_
# print("Feature Ranking: %s") % fit.ranking_
