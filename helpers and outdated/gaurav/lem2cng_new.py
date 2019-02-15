"""
    Script to generate new matrix(with new dimensions)
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
        print(_)
        for cng in vgs[_]:
            new_groups[_].append(cng)
    elif _ not in list(new_groups.keys()):
        new_groups[_] = vgs[_]
print('Collected New Noun Groups')
print('-' * 15)
print('New CNG Groups collected')
print('-' * 30)
#---------------build new groups json - END----------------#

mat_file = open('sample_lem2cng.json')
matrix = json.load(mat_file)
mat_file.close()
print('Loaded Old Matrix')
print('-' * 15)

# make a copy so or we can't loop through the matrix
updated_matrix = matrix.copy()
for lemma in matrix:
    print(rom_slp(lemma), 'THe RoMANov of', lemma)
    lemma = rom_slp(lemma)
    occurence_info = matrix[lemma]
    cngs = list(occurence_info.keys())
    for cng in cngs:
        for group in new_groups:
            if int(cng) in new_groups[group]:
                if group not in list(updated_matrix[lemma].keys()):
                    updated_matrix[lemma][group] = []
                [updated_matrix[lemma][group].append(f_name) for f_name in occurence_info[cng]]
    else:
        for group in new_groups:
            if int(cng) in new_groups[group]:
                if group not in list(updated_matrix[lemma].keys()):
                    updated_matrix[lemma][group] = []
                [updated_matrix[lemma][group].append(f_name) for f_name in occurence_info[cng]]

print("Final Updated Matrix")
for _ in updated_matrix:
    print("{", _, ":", updated_matrix[_], "}\n\n")
# print(new_groups)
print('-' * 30)

with open('lemma2cng_new.json', 'w') as lemma2cng:
    json.dump(updated_matrix, fp=lemma2cng)
