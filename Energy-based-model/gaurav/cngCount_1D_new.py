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

print('Collecting New CNG Groups')
print('-' * 30)
new_groups = {}
# get noun and verb group json
ng_file = open('noungroups.json')
ngs = json.load(ng_file)
new_groups.update(ngs)
ng_file.close()
print('Collected New Noun Groups')
print('-' * 15)

vg_file = open('verbgroups.json')
vgs = json.load(vg_file)
for vg in vgs:
    vgs[vg] = [(-1 * cng) for cng in vgs[vg]]
vg_file.close()
for _ in vgs:
    if _ in list(new_groups.keys()):
        print(_)
        for cng in vgs[_]:
            new_groups[_].append(cng)
    elif _ not in list(new_groups.keys()):
        new_groups[_] = vgs[_]
print('Collected New Noun Groups')
print('-' * 15)
# print(new_groups)
print('New CNG Groups collected')
print('-' * 30)


def CgsForCng(cng):
    cgs = []
    cgroups = list(new_groups.keys())
    for cg in cgroups:
        if cng in new_groups[cg]:
            cgs.append(cg)
    return cgs

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
    cng_list = []
    for arr in dcsObj.cng:
        cng_list += [rom_slp(c) for c in arr]
    cngCount = Counter(cng_list)
    # print(cngCount, len(cngCount))
    for cng in list(cngCount.keys()):

        # add cg counts
        CgsforCng = CgsForCng(int(cng))
        if len(CgsforCng) > 0:
            for cg in CgsforCng:
                if cg not in list(oneD.keys()):
                    oneD[cg] = 0
                oneD[cg] += cngCount[cng]

        # add cng counts
        if cng not in oneD:
            oneD[cng] = 0
        oneD[cng] += cngCount[cng]
with open('mat_cngCount_1D_complete.json', 'w') as f:
    json.dump(oneD, fp=f)
