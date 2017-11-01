import pickle
import numpy as np
import time
from multiprocessing import pool


def select_feature_on_threshold(feature_vector):
    return([feature_vector if np.unique(feature_vector).size > 150])

file_prefix = 'featureSet_500samples_8L_'
files_list = [(file_prefix + '_' + str(i) + '.p') for i in range(20)]

mat_load_time = 0
featureMatrix = np.array([])
for idx, featureFile in enumerate(files_list):
    start = time.time()
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')
    mat_load_time += time.time() - start
    if idx == 0:
        featureMatrix = df['featureMatrix']
        continue
    featureMatrix = np.concatenate((featureMatrix, df['featureMatrix']), axis=1)

print('Took %f seconds to load complete 10K_8L matrix' % mat_load_time)
with open('10K_8L.p', 'wb') as w:
    pickle.dump(featureMatrix, w)
print('Dumped to 10K_8L.p')


one_l_features = pool.map(feature_reduction_1L, featureMatrix)
print('Total time taken to reduce to 1L features from 8L using unique: %d' % total)
with open('1L_features_gs.p', 'wb') as f:
    pickle.dump(one_l_features, f)
