import pickle
import numpy as np

name = "featureSet_500samples_8L_"

files_list = []
for each in range(0, 20):
    files_list.append('featureSet_500samples_8L_'+str(each)+'.p')

featureMatrix = np.empty([106059, 9960])
all_pairs = []
targetDict = np.empty([9960, 4])

column1 = 0
current_column = 0

features = pickle.load(open("1L_feature.p", "rb"))

for featureFile in files_list:
    df = pickle.load(open(featureFile, 'rb'), encoding='utf8')

    current_column = len(df['all_pairs'])

    # get path lenghts for edge length 3
    for each in range(len(features)):
    	featureMatrix[each,column1:column1+current_column] = np.array(df['featureMatrix'][features[each]])


    for each in range(len(df['all_pairs'])):
        all_pairs.append(df['all_pairs'][each])

    temp_targetDict = []
    for each in range(len(df['targetDict'])):
        temp_targetDict.append(df['targetDict'][each])

    targetDict[column1:column1+current_column,:] = np.array(temp_targetDict)

    column1 += current_column
    #print(len(featureMatrix))
    #print(len(all_pairs))
    #print(len(targetDict))
    print(column1)

Dict1 = {}
Dict_all_pairs = {}
Dict_targetDict = {}
for each in range(len(all_pairs)):
	Dict_all_pairs[each] = all_pairs[each]
	Dict_targetDict[each] = targetDict[each]

Dict1['featureMatrix'] = featureMatrix
Dict1['all_pairs'] = Dict_all_pairs
Dict1['targetDict'] = Dict_targetDict

pickle.dump(Dict1,open("10K_1L.p","wb"),protocol=4)
