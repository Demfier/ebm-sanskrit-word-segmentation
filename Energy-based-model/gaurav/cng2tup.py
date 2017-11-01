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
mat_file = open('mat_tup2cng_new.json', 'r')
matrix = json.load(mat_file)

# mat_file = open('../Bishal/NewData/mat_tup2cng.p', 'rb')
# matrix = pickle.load(mat_file)
mat_file.close()
print('Loaded Old Matrix')
print('-' * 15)

updated_matrix = {}
ith_tup = 1
all_tups = len(list(matrix.keys()))
for tup in list(matrix.keys()):

    # To track progress
    print("%d/%d" % (ith_tup, all_tups))
    ith_tup += 1

    for cg in list(matrix[tup].keys()):
        if cg not in list(updated_matrix.keys()):
            updated_matrix[cg] = {}
        if tup not in list(updated_matrix[cg].keys()):
            updated_matrix[cg][tup] = []
        updated_matrix[cg][tup] += matrix[tup][cg]

for _ in updated_matrix:
    print(_, updated_matrix[_])
with open('mat_cng2tup_new.json', 'w') as f:
    json.dump(updated_matrix, fp=f)
