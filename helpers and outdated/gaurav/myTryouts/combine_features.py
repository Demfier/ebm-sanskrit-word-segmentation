import pickle
import os
import operator

files = os.listdir()

count = 0
for pkl in files:
    if pkl == 'combine_features.py' or pkl == 'features_combined':
        continue
    print("Processing %s" % pkl)
    features = pickle.load(open(pkl, 'rb'), encoding='utf8')
    long_list = []
    for feat_tup_list in features:
        long_list += feat_tup_list
    long_list.sort(reverse=True, key=operator.itemgetter(1))
    print(long_list[:100])
    filtered_feat_list = []
    encountered_indexs = []
    for tup in long_list:
        if tup[0] in encountered_indexs:
            continue
        filtered_feat_list.append(tup)
        encountered_indexs.append(tup[0])
    with open("combined_%d.p" % count, 'wb') as t:
        pickle.dump(filtered_feat_list, t)
    count += 1
