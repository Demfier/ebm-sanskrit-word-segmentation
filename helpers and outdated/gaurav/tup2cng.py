"""
    Script to generate new matrix(with new dimensions)
"""
import json
import pickle
from TxSeg_Summer.romtoslp import rom_slp

#---------------build new groups json - START----------------#
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
# this assumes that their is no key common to noungroups and verbgroups json
for _ in vgs:
    if _ in list(new_groups.keys()):
        print(_)
        for cng in vgs[_]:
            new_groups[_].append(cng)
    elif _ not in list(new_groups.keys()):
        new_groups[_] = vgs[_]
print('Collected New Noun Groups')
print('-' * 15)
print('New CNG Groups collected')
print('-' * 30)
# print(new_groups)
#---------------build new groups json - END----------------#

# For testing purposes
# mat_file = open('sample_tup2cng.json', 'r')
# matrix = json.load(mat_file)

mat_file = open('mat_tup2cng.p', 'rb')
# mat_file = open('../Bishal/NewData/mat_tup2cng.p', 'rb')
matrix = pickle.load(mat_file)
mat_file.close()
print('Loaded Old Matrix')
print('-' * 15)

updated_matrix = {}
# add cgs to the existing entries i.e {'lemma~cng': {'12_sg': [filenames]]}}
ith_tup = 1
all_tups = len(list(matrix.keys()))
for tup in list(matrix.keys()):

    # To track progress
    print("%d/%d" % (ith_tup, all_tups))
    ith_tup += 1

    lemmacng = tup.split('_')
    new_tup = ("%s_%s" % (rom_slp(lemmacng[0]), lemmacng[1]))
    updated_matrix[new_tup] = matrix[tup]
    current_cngs = list(matrix[tup].keys())
    for group in list(new_groups.keys()):
        updated_matrix[new_tup][group] = []
        for cng in current_cngs:
            cng = str(cng)
            if int(cng) in new_groups[group]:
                updated_matrix[new_tup][group] += matrix[tup][cng]
        if len(updated_matrix[new_tup][group]) == 0:
            del updated_matrix[new_tup][group]

# add cg as keys to the updated matrix i.e. {'lemma~cg': {'12_sg': [filenames]}}
# ith_tup = 1
# all_tups = len(list(matrix.keys()))
# for tup in list(matrix.keys()):
#
#     # To track progress
#     print("%d/%d" % (ith_tup, all_tups))
#     ith_tup += 1
#
#     lemmacng = tup.split('_')
#     lemma, cng = rom_slp(lemmacng[0]), lemmacng[1]
#     for gr in list(new_groups.keys()):
#         if int(cng) not in new_groups[gr]:
#             continue
#         n_tup = ("%s~%s" % (lemma, gr))
#         if n_tup not in list(updated_matrix.keys()):
#             updated_matrix[n_tup] = {}
#         for cg in list(updated_matrix[("%s~%s" % (lemma, cng))]):
#             if cg not in list(updated_matrix[n_tup].keys()):
#                 updated_matrix[n_tup][cg] = []
#             updated_matrix[n_tup][cg] += updated_matrix[("%s~%s" % (lemma, cng))][cg]

for _ in updated_matrix:
    print(_, updated_matrix[_].keys())
with open('mat_tup2cng_new.json', 'w') as f:
    json.dump(updated_matrix, fp=f)
