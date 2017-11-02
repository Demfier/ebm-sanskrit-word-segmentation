import word_definite as WD
from MatDB import *
import matplotlib.pyplot as plt
from IPython.display import display

import numpy as np
import math
np.set_printoptions(suppress=False, precision=16)

matDB = MatDB()

WD.word_definite_extInit(matDB)

print(len(matDB.mat_tupCount_1D))

smallTupList = []
for tup1, val in matDB.mat_tupCount_1D.items():
    if val > 20:
        smallTupList.append(tup1)      

l = len(smallTupList)
print(l)

batchCount = 2000
perm = np.random.permutation(l)[:batchCount]
# perm2 = np.random.permutation(l)[:batchCount]

fN = 4*443**2 + 9*443 + 9
print(fN)

all_pairs = {}

index = 0
for k in range(batchCount):  
    tup1 = smallTupList[perm[k]]
    lem1 = tup1.split('_')
    cng1 = lem1[1]
    lem1 = lem1[0]
    
    node1 = WD.word_definite(None, lem1, cng1, 0, 0)
    for tup2, co_occurrence in matDB.mat_tup2tup_countonly[tup1].items():
        if co_occurrence > 4:
            lem2 = tup2.split('_')
            cng2 = lem2[1]
            lem2 = lem2[0]
            node2 = WD.word_definite(None, lem2, cng2, 0, 1)
            all_pairs[index] = (node1, node2)
            index += 1
            
with open('outputs/log_001.txt', 'a') as log_handle:
    log_handle.write('Will get feature vectors for {} pairs\n'.format(index))
total_examples = index

pairs_per_file = 500

def tryForVal(mat, key1, key2):
    try:
        v = mat[key1][key2]
    except:
        v = 0
    return v

for pairx in range(math.ceil(len(all_pairs)/pairs_per_file)):
    subset_pairs = range(pairx*pairs_per_file, min(len(all_pairs), (pairx + 1)*pairs_per_file))
    featureMatrix = np.zeros((fN, len(subset_pairs)))
    targetDict = {}
    index = 0
    current_pairs = {}
    for hi in subset_pairs:
        node1 = all_pairs[hi][0]
        node2 = all_pairs[hi][1]
        current_pairs[index] = '{}^{}'.format(node1.tup, node2.tup)
        featureMatrix[:, index, None] = WD.Get_Features(node1, node2)
        targetDict[index] = (tryForVal(matDB.mat_tup2tup_countonly, node1.tup, node2.tup),\
                              tryForVal(matDB.mat_lem2lem_countonly, node1.lemma, node2.lemma),\
                              tryForVal(matDB.mat_lem2tup_countonly, node1.lemma, node2.tup),\
                              tryForVal(matDB.mat_tup2lem_countonly, node1.tup, node2.lemma))
        index += 1
        if index % min(math.ceil(pairs_per_file/2), 100) == 0:
            with open('outputs/log_001.txt', 'a') as log_handle:
                log_handle.write('Checkpoint S{}E{} of {}\n'.format(pairx, index, pairs_per_file))
    pickle.dump({'all_pairs': current_pairs, 'featureMatrix': featureMatrix, 'targetDict': targetDict},\
        open('outputs/featureSet_{}samples_8L_{}.p'.format(pairs_per_file, pairx), 'wb'), protocol = 4)
