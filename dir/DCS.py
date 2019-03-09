import sys
import warnings
from romtoslp import *
class DCS:
    def __init__(self,sent_id,sentence):
        self.sent_id=sent_id
        self.sentence=sentence
        self.dcs_chunks=[]
        self.lemmas=[]
        self.cng=[]

def SeeDCS(dcsObj):
    print('DCS ANALYZE')
    print('-'*15)
    print(dcsObj.sentence)
    print(dcsObj.lemmas)
    print("Lemmas:", [rom_slp(c) for arr in dcsObj.lemmas for c in arr])
    print(dcsObj.cng)
    print()