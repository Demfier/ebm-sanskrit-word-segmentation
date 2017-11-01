## bUILT-iN pACKAGES
import sys, os, csv
import pickle
from collections import defaultdict
import json
import numpy as np
import math
np.set_printoptions(suppress=True)
from IPython.display import display

## lAST sUMMER
from romtoslp import *
from sentences import *
from DCS import *
import MatDB
import time
import bz2
import zlib

## lAST yEAR
# from word_definite import *
# from nnet import *
# from heap_n_PrimMST import *
# from word_definite import *

is10K = False

if is10K:
    loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT_10K.p', 'rb'), encoding=u'utf-8')
    loaded_DCS = pickle.load(open('../Simultaneous_DCS_10K.p', 'rb'), encoding=u'utf-8')    
    outFolder = '../NewData/skt_dcs_DS.bz2_10K/'
else:
    loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT.p', 'rb'), encoding=u'utf-8')
    loaded_DCS = pickle.load(open('../Simultaneous_DCS.p', 'rb'), encoding=u'utf-8')
    outFolder = '../NewData/skt_dcs_DS.bz2/'

conversion_file_list = list(loaded_DCS.keys())
outFolder = '../NewData/skt_dcs_DS.bz2_1L_bigram_mir_Large/'

## SPECIAL - HELD OUT DATASET - OVERWRITES
'''
outFolder = '../NewData/skt_dcs_DS.bz2_1L_bigram_rfe_heldout/'
baseline_filelist = []
with open('inputs/Baseline4_advSample.csv') as f:
    baseline_reader = csv.reader(f)
    for line in baseline_reader:
        baseline_filelist.append(line[1])
        
conversion_file_list = [f.replace('.p', '.p2') for f in baseline_filelist]
#'''
## SPECIAL CODE ENDS HERE


dataset_4k_1k = pickle.load(open('../SmallDataset_4K_1K.p', 'rb'))
TrainFiles = dataset_4k_1k['TrainFiles']
TestFiles = dataset_4k_1k['TestFiles']

dataset_6k_3k = pickle.load(open('../SmallDataset_6K_3K.p', 'rb'))
TrainFiles_2 = dataset_6k_3k['TrainFiles']
TestFiles_2 = dataset_6k_3k['TestFiles']

matDB = MatDB.MatDB()

# from MatDB import *
import word_definite as WD
from heap_n_PrimMST import *
from nnet import *
"""
################################################################################################
######################  CREATE SEVERAL DATA STRUCTURES FROM SENTENCE/DCS  ######################
###########################  NODELIST, ADJACENCY LIST, GRAPH, HEAP #############################
################################################################################################
"""
def GetTrainingKit(sentenceObj, dcsObj):
    nodelist = GetNodes(sentenceObj)
    
    # Nodelist with only the correct_nodes
    nodelist2 = GetNodes(sentenceObj)
    nodelist2_to_correct_mapping = {}
    nodelist_correct = []
    search_key = 0
    first_key = 0
    for chunk_id in range(len(dcsObj.lemmas)):
        while nodelist2[first_key].chunk_id != chunk_id:
            first_key += 1
        for j in range(len(dcsObj.lemmas[chunk_id])):
            search_key = first_key
            while (nodelist2[search_key].lemma != rom_slp(dcsObj.lemmas[chunk_id][j])) or (nodelist2[search_key].cng != dcsObj.cng[chunk_id][j]):
                search_key += 1
                if search_key >= len(nodelist2) or nodelist2[search_key].chunk_id > chunk_id:
                    break
    #         print((rom_slp(dcsObj.lemmas[chunk_id][j]), dcsObj.cng[chunk_id][j]))
    #         print(nodelist[search_key])
            nodelist2_to_correct_mapping[len(nodelist_correct)] = search_key
            nodelist_correct.append(nodelist2[search_key])
    return (nodelist, nodelist_correct, nodelist2_to_correct_mapping)
    

def GetGraph(nodelist, neuralnet):
    if not neuralnet.outer_relu:
        conflicts_Dict = Get_Conflicts(nodelist)

        featVMat = Get_Feat_Vec_Matrix(nodelist, conflicts_Dict)

        (WScalarMat, SigmoidGateOutput) = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat, nodelist, conflicts_Dict, neuralnet)
        return (conflicts_Dict, featVMat, WScalarMat, SigmoidGateOutput)
    else:
        conflicts_Dict = Get_Conflicts(nodelist)

        featVMat = Get_Feat_Vec_Matrix(nodelist, conflicts_Dict)

        WScalarMat = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat, nodelist, conflicts_Dict, neuralnet)
        return (conflicts_Dict, featVMat, WScalarMat)
        
"""
################################################################################################
##############################  GET A FILENAME TO SAVE WEIGHTS  ################################
################################################################################################
"""
trainingStatus = defaultdict(lambda: bool(False))

class Trainer:
    def __init__(self):
        self.hidden_layer_size = 300
        self._edge_vector_dim = WD._edge_vector_dim
        # self._full_cnglist = list(WD.mat_cngCount_1D)
        self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True)
        self.history = defaultdict(lambda: list())

    def SaveToMem(self, sentenceObj, dcsObj, _debug = True):
        
        """ Pre-Process DCS and SKT to get all Nodes etc. """
        try:
            (nodelist, nodelist_correct, nodelist_to_correct_mapping) = GetTrainingKit(sentenceObj, dcsObj)
        except IndexError as e:
            # print('\x1b[31mError with {} \x1b[0m'.format(sentenceObj.sent_id))
            # print(e)
            return
        
#         startT = time.time()
        """ SKT FEATURE VECTOR MATRIX """
        conflicts_Dict_correct = Get_Conflicts(nodelist_correct)
        featVMat_correct = Get_Feat_Vec_Matrix(nodelist_correct, conflicts_Dict_correct)
        
        """ SKT FEATURE VECTOR MATRIX """
        conflicts_Dict = Get_Conflicts(nodelist)
        featVMat = Get_Feat_Vec_Matrix(nodelist, conflicts_Dict)
#         print('Nodelen: {}, Time taken to create: {}'.format(len(nodelist), time.time() - startT))
        
        with bz2.BZ2File(outFolder + sentenceObj.sent_id + '.ds.bz2', 'w') as f:
            pickle.dump({
                    'nodelist': nodelist,
                    'nodelist_correct': nodelist_correct,
                    'nodelist_to_correct_mapping': nodelist_to_correct_mapping,
                    'conflicts_Dict_correct': conflicts_Dict_correct,
                    'featVMat_correct': featVMat_correct,
                    'conflicts_Dict': conflicts_Dict,
                    'featVMat': featVMat
                }, f)
        
trainer = None
def InitModule(_matDB):
    global WD, trainer
    _edge_vec_dim = 1500
    WD.word_definite_extInit(_matDB, _edge_vec_dim)
    trainer = Trainer()
InitModule(matDB)
trainingStatus = defaultdict(lambda: bool(False))
# trainer.Load('outputs/train_nnet_t427031523027.p')

"""
################################################################################################
##############################  TRAIN FUNCTION  ################################################
################################################################################################
"""

def save_all_bz2(loaded_SKT, loaded_DCS, n_checkpt = 100):
    file_counter = 0
    print('{} files to process'.format(len(conversion_file_list)))
    for fn in conversion_file_list:
        if file_counter % n_checkpt == 0:
            print(file_counter,' Checkpoint... ')
            sys.stdout.flush() # Flush IO buffer 
        if os.path.isfile(outFolder + fn.replace('.p2', '.ds.bz2')):
            print('Skipping: ', fn)
            continue
        try:
            _ = trainer.SaveToMem(loaded_SKT[fn], loaded_DCS[fn])
        except (IndexError, KeyError) as e:
            pass
        file_counter += 1

if not os.path.isdir(outFolder):
    print('Creating directory: ', outFolder)
    os.mkdir(outFolder)
save_all_bz2(loaded_SKT, loaded_DCS, n_checkpt=100)
