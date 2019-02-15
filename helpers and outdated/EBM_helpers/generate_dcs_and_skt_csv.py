base_path_bz2 = '/home/rs/15CS91R05/Bishal/NewData/skt_dcs_DS.bz2_1L_bigram_heldout_dev/'

import operator
import bz2
import os
import pickle


# Function to open bz2 files (that contains both DCS & SKT info)
def open_dsbz2(filename):
    with bz2.BZ2File(filename, 'r') as f:
        loader = pickle.load(f)

    conflicts_Dict_correct = loader['conflicts_Dict_correct']
    nodelist_to_correct_mapping = loader['nodelist_to_correct_mapping']
    nodelist_correct = loader['nodelist_correct']
    featVMat_correct = loader['featVMat_correct']
    featVMat = loader['featVMat']
    conflicts_Dict = loader['conflicts_Dict']
    nodelist = loader['nodelist']

    return (nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
            nodelist, conflicts_Dict, featVMat)


bz2_files = os.listdir(base_path_bz2)
dcs_heldout_csv = ''
skt_heldout_csv = ''
count = 0
for ds in bz2_files:

    (nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
            nodelist, conflicts_Dict, featVMat) = open_dsbz2(base_path_bz2 + ds)

    fname = ds.replace('.ds.bz2', '')
    print(count, fname)
    count += 1

    lemmas = ''
    cngs = ''

    lemmas_dcs = ''
    cngs_dcs = ''

    for node in nodelist:
        lemmas += ',' + node.lemma
        cngs += ',' + node.cng
    lemmas += '\n'
    cngs += '\n'

    for node in nodelist_correct:
        lemmas_dcs += ',' + node.lemma
        cngs_dcs += ',' + node.cng
    lemmas_dcs += '\n'
    cngs_dcs += '\n'

    entry = fname + lemmas + fname + cngs
    entry_dcs = fname + lemmas_dcs + fname + cngs_dcs

    dcs_heldout_csv += entry_dcs
    skt_heldout_csv += entry

with open("final_task_gaurav/dcs_heldout.csv", "w") as f:
    f.write(dcs_heldout_csv)
with open("final_task_gaurav/skt_heldout.csv", "w") as f:
    f.write(skt_heldout_csv)
