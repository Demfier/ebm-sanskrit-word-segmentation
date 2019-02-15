import pickle
import numpy as np
import operator
from sklearn.feature_selection import mutual_info_regression
import math
import time


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


def filter_features(featureMatrix):
    filtered_matrix = []
    count = 0
    for idx, i in enumerate(featureMatrix):
        ## set to 150 for selecting ~20K features from 8L
        if len(set(i)) > 150:
            #print(idx)
            count += 1
            # print(len(set(i)), count, idx)
            filtered_matrix.append(i)
    print("Labels length", count)
    return(filtered_matrix)


def get_top_1500(filtered_matrix, all_pmi, all_pairs):
    X = []  # labels
    Y = []  # targets

    X = np.array(filtered_matrix).transpose()
    # build pmi array
    pmi = []
    for idx, pair in enumerate(all_pairs):
        pmi.append(all_pmi[pair])
    Y = np.array(pmi)
    assert len(X) == len(Y)
    # mir
    start = time.time()
    mi = mutual_info_regression(X, Y)
    # summarize the selection of the attributes
    print("time taken for mi")
    print(time.time() - start)

    listed_mi = list(mi)
    tupled_mi = []
    for idx, _ in enumerate(listed_mi):
        if str(_) != 'nan':
            tupled_mi.append(tuple((idx, _)))
    # sort by key's of tuple
    tupled_mi.sort(reverse=True, key=operator.itemgetter(1))
    if len(tupled_mi) > 4000:
        tupled_mi = tupled_mi[:4000]
    return(tupled_mi)

files_list = ['featureSet_500samples_8L_0.p', 'featureSet_500samples_8L_1.p', 'featureSet_500samples_8L_2.p', 'featureSet_500samples_8L_3.p', 'featureSet_500samples_8L_4.p', 'featureSet_500samples_8L_5.p', 'featureSet_500samples_8L_6.p', 'featureSet_500samples_8L_7.p', 'featureSet_500samples_8L_8.p', 'featureSet_500samples_8L_9.p', 'featureSet_500samples_8L_10.p', 'featureSet_500samples_8L_11.p', 'featureSet_500samples_8L_12.p', 'featureSet_500samples_8L_13.p', 'featureSet_500samples_8L_14.p', 'featureSet_500samples_8L_15.p', 'featureSet_500samples_8L_16.p', 'featureSet_500samples_8L_17.p', 'featureSet_500samples_8L_18.p', 'featureSet_500samples_8L_19.p']
files_list = ['10K_1L.p']

# featuresMatIndex = range(20)
for idx, featureFile in enumerate(files_list):
    print("Loading pickle file...")
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    featureMatrix = df['featureMatrix']
    all_pairs_dict = df['all_pairs']

    all_pairs = []
    for key in all_pairs_dict:
        all_pairs.append(all_pairs_dict[key])
    targetDict = df['targetDict']
    all_pmi = pickle.load(open('all_pmi.p', 'rb'), encoding='utf8')

    best_1500 = get_top_1500(filtered_matrix=featureMatrix, all_pmi=all_pmi, all_pairs=all_pairs)

    print("Dumping into mirCorr_for_%d single file" % idx)
    with open('mirpmi_for_%d.p' % idx, 'wb') as mifs:
        pickle.dump(best_1500, mifs)
