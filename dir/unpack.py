from Train_clique import *
from heap_n_clique import *
from nnet import *
from TestPool_Unit_clique import *
from sentences import *

bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_bigram_mir_10K/'  #bm2
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_bigram_mir_10K/'   #bm3
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_bigram_rfe_10K/'   #br2
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_bigram_rfe_10K/'   #br3
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_pmi_mir_10K/'   #pm2
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_pmi_mir_10K2/'   #pm3
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_pmi_rfe_10K/'   #pr2
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_pmi_rfe_10K/'   #pr3
loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT_10K.p', 'rb'), encoding=u'utf-8')
loaded_DCS = pickle.load(open('../Simultaneous_DCS_10K.p', 'rb'), encoding=u'utf-8')

dsbz2_name = '4442.ds.bz2'

(nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
            nodelist, conflicts_Dict, featVMat) = open_dsbz2(bz2_input_folder + dsbz2_name)

# print(nodelist_correct)
# print(nodelist)



sentenceObj = loaded_SKT['4442.p2']

# SeeSentence(sentenceObj)
WScalarMat_correct = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat_correct, nodelist_correct,\
                                                                      conflicts_Dict_correct, self.neuralnet)
source = 0

(min_st_gold_ndict, min_st_adj_gold_small, _) =MST(nodelist_correct, WScalarMat_correct, conflicts_Dict_correct, source)
energy_gold_max_ST = np.sum(WScalarMat_correct[min_st_adj_gold_small])


print(min_st_gold_ndict)
