import pickle
import numpy as np
import operator
from sklearn.feature_selection import chi2, SelectKBest
import time

# featuresMatIndex = range(20)
start = time.time()
# for findex in featuresMatIndex:
featureFile = 'featureSet_500samples_8L_19.p'
print("Loading pickle file...")
df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
featureMatrix = df['featureMatrix']
targetDict = df['targetDict']
print("finished loading pickel file and built feature and target matrix.")
print(len(targetDict))
print("Took", time.time() - start, "seconds")


def get_top_500(targetIndex, featureMatrix=featureMatrix, targetDict=targetDict):
    X = []  # labels
    Y = []  # targets
    for idx, i in enumerate(featureMatrix.transpose()):
        if len(set(i)) > 100:
            X.append(i)
            # set to lemma2lemma
            Y.append(targetDict[idx][targetIndex])
    assert len(X) == len(Y)
    print(len(X))
    # univariate
    test = SelectKBest(score_func=chi2, k=500)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    print("Final fitting scores")
    print(fit.scores_)

    listed_kbest = list(fit.scores_)
    tupled_kbest = []
    for idx, _ in enumerate(listed_kbest):
        if str(_) != 'nan':
            tupled_kbest.append(tuple((idx, _)))
        else:
            tupled_kbest.append(tuple((idx, -0.00001)))

    # sort by key's of tuple
    tupled_kbest.sort(reverse=True, key=operator.itemgetter(1))
    print("top 500 features are")
    print(tupled_kbest[:500])
    return(tupled_kbest[:500])

top_500_kbest = []
for i in range(4):
    print("%d/4 iteration" % (i + 1))
    print("getting top 500 features for this feature")
    best_500 = get_top_500(i)
    print("Extraction complete.")
    print("Took", time.time() - start, "seconds")
    top_500_kbest.append(best_500)

print("Dumping into single file")
with open('kbest_results_final_19.p', 'wb') as kbest:
    pickle.dump(top_500_kbest, kbest)
