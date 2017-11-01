import pickle
import numpy as np
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
#
print("Loading pickle file...")
# print("Starting reading features")
# for featVec in featureMatrix:


# from pandas import read_pickle
# load data
# names = range(500)
X = []
Y = []
df = pickle.load(open('featureSet_500samples_8L_0.p', 'rb'), encoding='utf8')
featureMatrix = df['featureMatrix']
targetDict = df['targetDict']
print(len(targetDict))
for idx, i in enumerate(featureMatrix.transpose()):
    if len(set(i)) > 80:
        X.append(i)
        # set to word2word
        Y.append(targetDict[idx][0])
print("finished loading pickel file.")
print(len(X), len(Y))
# RFE
model = LogisticRegression()
rfe = RFE(model, 500)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

# univariate
# test = SelectKBest(score_func=chi2, k=10)
# fit = test.fit(X, Y)
# # summarize scores
# np.set_printoptions(precision=3)
# print(fit.scores_)
with open('rfe_results.p', 'wb') as rfe:
    pickle.dump(fit.support_, rfe)
# features = fit.transform(X)
