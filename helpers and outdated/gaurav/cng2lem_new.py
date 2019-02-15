"""
    Script to generate new lemma2cng mtrix(with new dimensions)
"""
import json
import time
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
        for cng in vgs[_]:
            new_groups[_].append(cng)
    elif _ not in list(new_groups.keys()):
        new_groups[_] = vgs[_]
print('Collected New Noun Groups')
print('-' * 15)
print('New CNG Groups collected')
print('-' * 30)
#---------------build new groups json - END----------------#
# print(new_groups)

mat_file = open('sample_cng2lem.json')
matrix = json.load(mat_file)
mat_file.close()
print('Loaded Old Matrix')
print('-' * 15)

# make a copy so or we can't loop through the matrix
updated_matrix = matrix.copy()
print(list(matrix.keys()))
print(new_groups)
for group in list(new_groups.keys()):
    if group not in list(updated_matrix.keys()):
        updated_matrix[group] = {}
    for cng in new_groups[group]:
        if str(cng) not in list(matrix.keys()):
            continue
        print(cng)
        for lemma in matrix[str(cng)]:
            print(rom_slp(lemma), 'THe RoMANov of', lemma)
            if lemma not in list(updated_matrix[group].keys()):
                updated_matrix[group][lemma] = []
                print("created lemma", lemma, "for cng", group)
            updated_matrix[group][lemma] += matrix[str(cng)][lemma]
    if not updated_matrix[group]:
        del updated_matrix[group]

for _ in updated_matrix:
    print(_, updated_matrix[_], '\n\n\n')

with open('cng2lemma_new.json', 'w') as cng2lemma:
    json.dump(updated_matrix, fp=cng2lemma)
