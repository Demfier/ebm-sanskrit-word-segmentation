"""
    Script to generate new matrix(with new dimensions)
"""
import json
import _pickle as cpickle
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

mat_file = open('data7.json')
matrix = json.load(mat_file)
mat_file.close()
print('Loaded Old Matrix')
print('-' * 15)

# make a copy so or we can't loop through the matrix
updated_matrix = matrix.copy()
# print(matrix)

# To add cg to cngs
for cg in list(matrix.keys()):
    for cng in list(matrix[cg].keys()):
        for group in list(new_groups.keys()):
            if int(cng) not in new_groups[group]:
                continue
            if group not in list(updated_matrix[cg].keys()):
                updated_matrix[cg][group] = []
            updated_matrix[cg][group] += matrix[cg][cng]

# To add cg to the matrix for queries like matrix['12_sp']['nom']
for group in list(new_groups.keys()):
    updated_matrix[group] = {}
    for cng in list(matrix.keys()):
        if group in list(updated_matrix[cng].keys()):
            updated_matrix[group][cng] = updated_matrix[cng][group]
    current_cngs = list(updated_matrix[group].keys())
    for cg in list(new_groups.keys()):
        cg = str(cg)
        updated_matrix[group][cg] = []
        for cng in current_cngs:
            cng = str(cng)
            if int(cng) in new_groups[cg]:
                updated_matrix[group][cg] += updated_matrix[group][cng]
        if len(updated_matrix[group][cg]) == 0:
            del updated_matrix[group][cg]
    if len(updated_matrix[group]) == 0:
        del updated_matrix[group]

for _ in updated_matrix:
    print(_, updated_matrix[_].keys())
with open('cng2cng_with_cgs.p', 'wb') as f:
    cpickle.dump(updated_matrix, f)
