#from Train_n_Save_NNet import *
import bz2,pickle
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
    print("conflicts_Dict_correct:",)
    print(conflicts_Dict_correct)
    print("nodelist_to_correct_mapping: ")
    print(nodelist_to_correct_mapping)

    print("nodelist_correct")
    nc0 = nodelist_correct[0]
    print(type(nc0))
    print(nodelist_correct[0])

    print("featVMat_correct")
    print(featVMat_correct[0][1][0])
	
    print("featVMat")
    print(featVMat[0][0])

    print("conflicts_Dict")
    print(conflicts_Dict)
    
    print("nodelist")
    print(nodelist)
    #return (nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,nodelist, conflicts_Dict, featVMat)

print(open_dsbz2("100004.ds.bz2"))
