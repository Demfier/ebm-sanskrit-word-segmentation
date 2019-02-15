import os	

testfolder='../NewData/skt_dcs_DS.bz2_4K_bigram_mir_heldout'
print('loading Testing FIles')
TestFiles=set()
Allfiles=set()
for f in os.listdir(testfolder):
    if '.ds.bz2' in f:
    	f = f.replace('.ds.bz2', '.p2')
        TestFiles.add(f)
        Allfiles.add((f,1))
bz2_input_folder = '../NewData/skt_dcs_DS.bz2_4K_bigram_mir_10K/'
print('loading Training Files')
TrainFiles = set()
for f in os.listdir(bz2_input_folder):
    if '.ds.bz2' in f:
    	f = f.replace('.ds.bz2', '.p2')
        TrainFiles.add(f)
        Allfiles.add((f,2))



# TestFiles = sorted(TestFiles)
# print(TestFiles)

# print

# TrainFiles = sorted(TrainFiles)
# print(TrainFiles)

print(TestFiles&TrainFiles)


print(len(TrainFiles))
print(len(TestFiles))
# for i in sorted(Allfiles):
# 	print i