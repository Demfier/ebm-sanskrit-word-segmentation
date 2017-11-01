from collections import Counter
import pickle
import os
from TxSeg_Summer.romtoslp import rom_slp
import json


class DCS:
    def __init__(self, sent_id, sentence):
        self.sent_id = sent_id
        self.sentence = sentence
        self.dcs_chunks = []
        self.lemmas = []
        self.cng = []


def pickleFixLoad(filename):
    # open file as binary ('rb') and encoding = u'utf8' argument is necessary
    return pickle.load(open(filename, 'rb'), encoding=u'utf8')

dcsPath = "../../Bishal/Text Segmentation/DCS_pick/"
print("Loading DCS files")
dcsList = os.listdir(dcsPath)
print("Loaded all %d DCS files" % len(dcsList))
# print(dcsList[:2])

oneD = {}
total = len(dcsList)
u = 0
for dcsFile in dcsList:
    print("%d/%d" % (u + 1, total))
    u += 1
    try:
        dcsObj = pickleFixLoad(dcsPath + dcsFile)
    except pickle.UnpicklingError as e:
        print("Okay some thing wrong with the file %s" % dcsFile)
    except EOFError:
        print("Okay some thing wrong with the file %s" % dcsFile)

    # get lemmas
    lemma_list = []
    for arr in dcsObj.lemmas:
        lemma_list += [rom_slp(l) for l in arr]
    lemmaCount = Counter(lemma_list)

    # get cngs
    cng_list = []
    for arr in dcsObj.cng:
        cng_list += [c for c in arr]
    cngCount = Counter(cng_list)

    # add tup to the matrix
    for lemma in list(lemmaCount.keys()):
        for cng in list(cngCount.keys()):
            tup = lemma + '_' + cng
            if tup not in list(oneD.keys()):
                oneD[tup] = 0
            oneD[tup] += lemmaCount[lemma]

with open('mat_tupCount_1D.json', 'w') as f:
    json.dump(oneD, fp=f)
