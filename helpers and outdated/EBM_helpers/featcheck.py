import os
import sys,bz2,pickle


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


# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_bigram_mir_10K/'  #bm2  2000
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_bigram_mir_10K/'   #bm3  1500
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_bigram_rfe_10K/'   #br2  2000
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_bigram_rfe_10K/'   #br3  2000
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_pmi_mir_10K/'   #pm2   1500
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_pmi_mir_10K2/'   #pm3  2000
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_pmi_rfe_10K/'   #pr2   2000
# bz2_input_folder = '../NewData/skt_dcs_DS.bz2_1L_pmi_rfe_10K/'   #pr3   2000

ii=0
for f in os.listdir(bz2_input_folder):
	try:    
		ii+=1
		if(ii>10):
			break
		print(bz2_input_folder)
		print(f)
		(nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
				nodelist, conflicts_Dict, featVMat) = open_dsbz2(bz2_input_folder + f)
		print(len(nodelist))
		nodesCount = len(nodelist)
		# print(len(featVMat[0][6]))
		# print(featVMat[0][6][0][0])
		for i in range(nodesCount):
			for j in range(nodesCount):
				try:
					print("***"*20)
					print(len(featVMat[i][j]))
					print("---"*20)
					print(featVMat[i][j][len(featVMat[i][j])-1])
					if(featVMat[i][j][0]!=featVMat[i][j][len(featVMat[i][j])-1]):
						print("!?"*30)
						print(f)
						print(i)
						print(j)
					print("###"*20)
				except:
					continue
	except:
		continue


# f="389979.ds.bz2"
# (nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
# 				nodelist, conflicts_Dict, featVMat) = open_dsbz2(bz2_input_folder + f)
# print(sorted(featVMat[21][15])[0:1500][0])

# print(max(featVMat[21][15]))