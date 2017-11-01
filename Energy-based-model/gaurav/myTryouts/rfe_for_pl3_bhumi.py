"""Will produce ranking_x.p file and rfe_top1500_for_x.p"""

import pickle
import numpy as np
import operator
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
import time

def get_top_1500(featureMatrix, pmiDict, all_pairs):
    X = []  # labels
    Y = []  # targets
    count = 0
    filtered_matrix = []

    for idx, i in enumerate(featureMatrix):
        count += 1
        #print(len(set(i)), count, idx)
        filtered_matrix.append(i)

    X = np.array(filtered_matrix).transpose()
    # build targets array
    pmi = []
    for idx, tup in enumerate(all_pairs):
    	#print(tup)
    	pmi.append((pmiDict[tup]))
    Y = np.array(pmi)
    print("Labels length", count)
    assert len(X) == len(Y)
    # RFE
    print("Building Regression Model using RFE")
    #model = LogisticRegression()
    #model = SVR(kernel='linear')
    #model = LinearRegression()
    model = mutual_info_regression(X,Y)
    print(model.coef_)
    rfe = RFE(model, 1500,step = 20)
    fit = rfe.fit(X, Y)
    
    # summarize the selection of the attributes
    print(fit.support_)

    listed_rfe = list(fit.support_)
    tupled_rfe = []
    for idx, _ in enumerate(listed_rfe):
        if _:
            tupled_rfe.append(tuple((idx, _)))

    # get ranking of features
    ranking = []
    for idx, _ in enumerate(fit.ranking_):
        ranking.append(tuple((idx, _)))

    with open('ranking_pmi.p', 'wb') as r:
        pickle.dump(ranking, r)
    return(tupled_rfe)

# 10K_1L.p contains around 10K datapoints and 1L features which are selected with thresold 150
files_list = ['Bhumi/10K_4K.p']


for featureFile in files_list:
    # featuresMatIndex = range(20)
    start = time.time()
    print("Loading %s file..." % featureFile)
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    pmiDict = pickle.load(open('all_pmi.p', 'rb'), encoding='utf8')
    
    # get only 4K features 
    featureMatrix=df['featureMatrix']
    
    all_pairs = df['all_pairs']

    print("finished loading pickle file and built feature and target matrix.")
    print("Took", time.time() - start, "seconds")

    top_1500_rfe = []
    print("getting top 1500 features for this feature")
    best_1500 = get_top_1500(featureMatrix, pmiDict, all_pairs)
    print("Extraction complete.")
    print("Took", time.time() - start, "seconds")
    top_1500_rfe.append(best_1500)
    
    print("Dumping into single file")
    with open('rfe_mirpmi_for_%s' % featureFile, 'wb') as rfe:
        pickle.dump(top_1500_rfe, rfe)


