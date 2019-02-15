import os

folders = ['skt_dcs_DS.bz2_1L_bigram_mir_Large']

for folder in folders:
	path = os.path.join('../NewData', folder)
	c = len(os.listdir(path))
	print('Folder: {:35s} ------> File_Count: {}\n'.format(folder, c))
