import pickle
import numpy as np
import operator
from mifs import mifs
import time


def filter_features(featureMatrix):
    filtered_matrix = []
    count = 0
    for idx, i in enumerate(featureMatrix):
        ## set to 370 for selecting ~20K features from 8L
        if len(set(i)) > 370:
            count += 1
            # print(len(set(i)), count, idx)
            filtered_matrix.append(i)
    print("Labels length", count)
    return(filtered_matrix)


def get_top_500(targetIndex, filtered_matrix, targetDict):
    X = []  # labels
    Y = []  # targets
    # for idx, i in enumerate(featureMatrix):
    #     ## set to 370 for selecting ~20K features from 8L
    #     if len(set(i)) > 370:
    #         count += 1
    #         # print(len(set(i)), count, idx)
    #         filtered_matrix.append(i)

    X = np.array(filtered_matrix).transpose()
    # build targets array
    for idx in range(len(targetDict)):
        Y.append(targetDict[idx][targetIndex])
    assert len(X) == len(Y)
    # RFE
    feat_selector = mifs.MutualInformationFeatureSelector(k=0)
    feat_selector.fit(X, Y)

    listed_mi = list(feat_selector.support_)
    tupled_mi = []
    for idx, _ in enumerate(listed_mi):
        if str(_) != 'nan':
            tupled_mi.append(tuple((idx, _)))

    # sort by key's of tuple
    tupled_mi.sort(reverse=True, key=operator.itemgetter(1))
    print("top 500 features are")
    print(tupled_mi[:500])
    return(tupled_mi[:500])

files_list = ['featureSet_500samples_8L_0.p', 'featureSet_500samples_8L_1.p', 'featureSet_500samples_8L_2.p', 'featureSet_500samples_8L_3.p', 'featureSet_500samples_8L_4.p', 'featureSet_500samples_8L_5.p', 'featureSet_500samples_8L_6.p', 'featureSet_500samples_8L_7.p', 'featureSet_500samples_8L_8.p', 'featureSet_500samples_8L_9.p', 'featureSet_500samples_8L_10.p', 'featureSet_500samples_8L_11.p', 'featureSet_500samples_8L_12.p', 'featureSet_500samples_8L_13.p', 'featureSet_500samples_8L_14.p', 'featureSet_500samples_8L_15.p', 'featureSet_500samples_8L_16.p', 'featureSet_500samples_8L_17.p', 'featureSet_500samples_8L_18.p', 'featureSet_500samples_8L_19.p']

# featuresMatIndex = range(20)
for idx, featureFile in enumerate(files_list):

    start = time.time()
    # for findex in featuresMatIndex:

    # Loading feature pickle
    print("Loading pickle file...")
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    featureMatrix = df['featureMatrix']
    targetDict = df['targetDict']
    print("finished loading pickel file and built feature and target matrix.")
    print(len(targetDict))
    print("Took", time.time() - start, "seconds")

    # filter features
    filtered_matrix = filter_features(featureMatrix)

    # Get top 500 features
    top_500_mifs = []
    for i in range(4):
        print("%d/4 iteration" % (i + 1))
        print("getting top 500 features for this feature")
        best_500 = get_top_500(i, filtered_matrix=filtered_matrix, targetDict=targetDict)
        print("Extraction complete.")
        print("Took", time.time() - start, "seconds")
        top_500_mifs.append(best_500)

    print("Dumping into single file")
    with open('mifs_results_final_%d.p' % idx, 'wb') as mifs:
        pickle.dump(top_500_mifs, mifs)
