
# coding: utf-8

# In[1]:

import pandas,sys


# In[7]:
fils = {
'BM3' : ["BM3_NLoss_proc0.csv","BM3_NLoss_proc2.csv","BM3_NLoss_proc1.csv","BM3_NLoss_proc3.csv"],
'BM2' : ["BM2_NLoss_proc0.csv","BM2_NLoss_proc2.csv","BM2_NLoss_proc1.csv","BM2_NLoss_proc3.csv"],
'BR2' : ["BR2_NLoss_proc0.csv","BR2_NLoss_proc2.csv","BR2_NLoss_proc1.csv","BR2_NLoss_proc3.csv"],
'BR3' : ["BR3_NLoss_proc0.csv","BR3_NLoss_proc2.csv","BR3_NLoss_proc1.csv","BR3_NLoss_proc3.csv"],
'PM2' : ["PM2_NLoss_proc0.csv","PM2_NLoss_proc2.csv","PM2_NLoss_proc1.csv","PM2_NLoss_proc3.csv"],
'PM3' : ["PM3_NLoss_proc0.csv","PM3_NLoss_proc2.csv","PM3_NLoss_proc1.csv","PM3_NLoss_proc3.csv"],
'PR2' : ["PR2_NLoss_proc0.csv","PR2_NLoss_proc2.csv","PR2_NLoss_proc1.csv","PR2_NLoss_proc3.csv"],
'PR3' : ["PR3_NLoss_proc0.csv","PR3_NLoss_proc2.csv","PR3_NLoss_proc1.csv","PR3_NLoss_proc3.csv"]
}   
import pandas
from collections import defaultdict

def predLoss(tag):
    gt = defaultdict(dict)

    for item in fils[tag]:
        fil = open('outputs/'+str(item)).read().splitlines()
        for i,line in enumerate(fil):
            if i % 6 == 0:
                setCol = line.split(',')
                gt[setCol[0]]['predLemma'] = setCol[1:]
            if i%6 == 1:
                gt[setCol[0]]['predCNG'] = line.split(',')[1:]
                if len(gt[setCol[0]]['predLemma']) != len(gt[setCol[0]]['predCNG']):
                    print(gt[setCol[0]])
            if i%6 == 2:
                gt[setCol[0]]['chunkID'] = line.split(',')[1:]
                if len(gt[setCol[0]]['predLemma']) != len(gt[setCol[0]]['chunkID']):
                    print(gt[setCol[0]])
            if i%6 == 3:
                gt[setCol[0]]['chunkIDCNG'] = line.split(',')[1:]
                if len(gt[setCol[0]]['predLemma']) != len(gt[setCol[0]]['chunkIDCNG']):
                    print(gt[setCol[0]])
            if i%6 == 4:
                gt[setCol[0]]['idInNodeID'] = line.split(',')[1:]
                if len(gt[setCol[0]]['predLemma']) != len(gt[setCol[0]]['idInNodeID']):
                    print(gt[setCol[0]])
            if i%6 == 5:
                gt[setCol[0]]['params'] = line.split(',')[1:]

            if line.split(',')[0] != setCol[0]:
                print(i,setCol,line)
                print('breakin')
                break
    return gt

def pdframe(gt):
    params = defaultdict(dict)
    for item in gt.keys():
        tatkal = gt[item]['params']
        params[item]['corrWords'],params[item]['corrLemma'] = int(tatkal[0]),int(tatkal[1])
        params[item]['dcsSize'],params[item]['predictions'] = int(tatkal[2]),int(tatkal[3])
        params[item]['word++Precision'] = params[item]['corrWords']*1.0/params[item]['predictions']
        params[item]['word++Recall'] = params[item]['corrWords']*1.0/params[item]['dcsSize']
        params[item]['wordPrecision'] = params[item]['corrLemma']*1.0/params[item]['predictions']
        params[item]['wordRecall'] = params[item]['corrLemma']*1.0/params[item]['dcsSize']


    initRes = pandas.DataFrame.from_dict(params,orient='index')
    return initRes


# In[8]:

if(len(sys.argv)<2):
    print("Provide an argument for the feature to be evaluated")

else:
    BM2gt = predLoss(str(sys.argv[1]))
    BM2pd = pdframe(BM2gt)
    print(BM2pd.mean())