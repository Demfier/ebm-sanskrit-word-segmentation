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
    filtered_id = []
    for idx, i in enumerate(featureMatrix):
        ## set to 370 for selecting ~20K features from 8L
        count += 1
        if len(set(i)) > 150:
            #print(idx)
            #count += 1
            # print(len(set(i)), count, idx)
            filtered_matrix.append(i)
            filtered_id.append(idx)
    print("Labels length", count)
    return(filtered_matrix,filtered_id)

start = time.time()


def get_top_500(filtered_matrix, all_pmi, all_pairs):
    X = []  # labels
    Y = []  # targets
    # count = 0
    # for idx, i in enumerate(featureMatrix):
    #     ## set to 370 for selecting ~20K features from 8L
    #     if len(set(i)) > 370:
    #         count += 1
    #         # print(len(set(i)), count, idx)
    #         filtered_matrix.append(i)

    X = np.array(filtered_matrix).transpose()
    # build pmi array
    pmi = []
    for idx, pair in enumerate(all_pairs):
        pmi.append(all_pmi[pair])
    print("pmi vector")
    print(pmi)
    Y = np.array(pmi)
    assert len(X) == len(Y)
    # mir
    mi = mutual_info_regression(X, Y)
    # summarize the selection of the attributes
    print(mi)
    print("time taken for mi")
    print(time.time() - start)

    listed_mi = list(mi)
    tupled_mi = []
    for idx, _ in enumerate(listed_mi):
        if str(_) != 'nan':
            tupled_mi.append(tuple((idx, _)))

    # sort by key's of tuple
    tupled_mi.sort(reverse=True, key=operator.itemgetter(1))
    print("top 20K features are")
    print(tupled_mi)
    if len(tupled_mi) > 20000:
        tupled_mi = tupled_mi[:20000]

    featureVec = {}
    for tup in tupled_mi:
        featureVec[tup[0]] = filtered_matrix[tup[0]]
    print(len(featureVec))
    corr = []
    for idx, key in enumerate(list(featureVec.keys())):
        corr.append(tuple((key, pearson_def(featureVec[key], pmi))))
    corr.sort(reverse=True, key=operator.itemgetter(1))
    print("top 600 features are")
    return(corr[:1500])

files_list = ['featureSet_500samples_8L_0.p', 'featureSet_500samples_8L_1.p', 'featureSet_500samples_8L_2.p', 'featureSet_500samples_8L_3.p', 'featureSet_500samples_8L_4.p', 'featureSet_500samples_8L_5.p', 'featureSet_500samples_8L_6.p', 'featureSet_500samples_8L_7.p', 'featureSet_500samples_8L_8.p', 'featureSet_500samples_8L_9.p', 'featureSet_500samples_8L_10.p', 'featureSet_500samples_8L_11.p', 'featureSet_500samples_8L_12.p', 'featureSet_500samples_8L_13.p', 'featureSet_500samples_8L_14.p', 'featureSet_500samples_8L_15.p', 'featureSet_500samples_8L_16.p', 'featureSet_500samples_8L_17.p', 'featureSet_500samples_8L_18.p', 'featureSet_500samples_8L_19.p']
files_list = ['featureSet_500samples_8L_0.p']

# featuresMatIndex = range(20)
for idx, featureFile in enumerate(files_list):

    # for findex in featuresMatIndex:

    # Loading feature pickle
    print("Loading pickle file...")
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    featureMatrix = df['featureMatrix']
    all_pairs_dict = df['all_pairs']

    all_pairs = []
    for key in all_pairs_dict:
        all_pairs.append(all_pairs_dict[key])
    # print(all_pairs)
    targetDict = df['targetDict']
    print("finished loading pickle file and built feature and target matrix.")
    print(len(targetDict))
    print("Took", time.time() - start, "seconds")

    all_pmi = pickle.load(open('all_pmi.p', 'rb'), encoding='utf8')
    # filter features
    [filtered_matrix,filtered_id] = filter_features(featureMatrix)

    print(len(filtered_id))
    #pickle.dump(filtered_id,open("1p","wb"))

    
    # Get top 500 features
    print("getting top 500 features for this feature")
    best_500 = get_top_500(filtered_matrix=filtered_matrix, all_pmi=all_pmi, all_pairs=all_pairs)
    print("Extraction complete.")
    print("Took", time.time() - start, "seconds")
    '''
    print("Dumping into mirCorr_for_%d single file" % idx)
    with open('mirCorr_for_%d.p' % idx, 'wb') as mifs:
        pickle.dump(best_500, mifs)
    '''
