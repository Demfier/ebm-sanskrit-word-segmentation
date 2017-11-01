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


def SeeDCS(dcsObj):
    print('DCS ANALYZE')
    print('-' * 15)
    print(dcsObj.sentence)
    print(dcsObj.lemmas)
    print("Lemmas:", [rom_slp(c) for arr in dcsObj.lemmas for c in arr])
    print(dcsObj.cng)

dcsPath = "../../Bishal/Text Segmentation/DCS_pick/"
print("Loading DCS files")
dcsList = os.listdir(dcsPath)
# print(dcsList[:2])

oneD = {}

for dcsFile in dcsList:
    try:
        dcsObj = pickleFixLoad(dcsPath + dcsFile)
    except pickle.UnpicklingError as e:
        print("Okay some thing wrong with the file %s" % dcsFile)
    except EOFError:
        print("Okay some thing wrong with the file %s" % dcsFile)
    lemma_list = []
    for arr in dcsObj.lemmas:
        lemma_list += [rom_slp(l) for l in arr]
    lemmaCount = Counter(lemma_list)
    # print(lemmaCount, len(lemmaCount))
    for lemma in list(lemmaCount.keys()):
        lemma = rom_slp(lemma)
        if lemma not in oneD:
            oneD[lemma] = 0
        oneD[lemma] += lemmaCount[lemma]
with open('mat_lemCount_1D.json', 'w') as f:
    json.dump(oneD, fp=f)
