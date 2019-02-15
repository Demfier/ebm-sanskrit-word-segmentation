import pandas as pd
import numpy as np
import csv,pickle,json,bz2
from romtoslp import *
loaded_DCS = pickle.load(open('../Simultaneous_DCS_ho.p', 'rb'))
folder =  '../NewData/skt_dcs_DS.bz2_4K_bigram_mir_heldout/'

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

#snippet for forming the groundtruth csv file
with open('groundtruth2.csv','w') as fh:
    rd = csv.writer(fh)
    rd.writerow(['File','Lemma','CNG','lemmaCorr','lemmaCNGcorr','predCNG','Conflicts'])
count=0
for ii in range(4): 
    with open("BM2_NLoss_proc"+str(ii)+".csv",'r') as fh:
        rd = csv.reader(fh)
        while(True):
            try:
                print(count)
                count+=1
                x=next(rd)  #predicted lemmas
                sentid = x[0]
                dcsobj = loaded_DCS[str(sentid)+'.p2']
#                 print(dcsobj.cng)
#                 print(dcsobj.lemmas)
#                 print(dcsobj.dcs_chunks)
                nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
            nodelist, conflicts_Dict, featVMat = open_dsbz2(folder+str(sentid)+'.ds.bz2')
#                 print(conflicts_Dict_correct)
#                 print(nodelist_correct)
#                 break
                dll = 0
                for i in dcsobj.lemmas:
                    dll+=len(i)
                if(dll!=len(nodelist_correct)):
                    print('here')
                    print(dcsobj.lemmas)
                    print(nodelist_correct)
                gtlemmas = []
                for outerlist in dcsobj.lemmas:
                    for element in outerlist:
                        gtlemmas.append(rom_slp(element))
                pdlemmas = x[1:]
                
                x=next(rd) #predicted cngs
                gtcngs = []
                i = 0
                for outerlist in dcsobj.cng:
                    for element in outerlist:
                        gtcngs.append((element,len(conflicts_Dict_correct[i])))
                        i+=1
                pdcngs = x[1:]
                for i in range(4):
                    x=(next(rd))
#                 print(gtlemmas)
#                 print(pdlemmas)
#                 print(gtcngs)
#                 print(pdcngs)               
                pdldict = dict()
                gtldict = dict()
                for i in range(len(gtlemmas)):
                    if(gtlemmas[i] in gtldict):
                        gtldict[gtlemmas[i]].append(gtcngs[i])
                    else:
                        gtldict[gtlemmas[i]] = [gtcngs[i]]
                
                for i in range(len(pdlemmas)):
                    if(pdlemmas[i] in pdldict):
                        pdldict[pdlemmas[i]].append(pdcngs[i])
                    else:
                        pdldict[pdlemmas[i]] = [pdcngs[i]]
                
#                 print(gtldict)
#                 print(gtldict)
            
                lemmaround2 = []
                cnground2 = []
                for gtl in gtldict.keys():                  
                    for gtlcng in gtldict[gtl]:                     
                        lemmacorr = 0
                        lemmaCNGcorr=0
                        predictedcng = 'nil'
                        confcount = gtlcng[1]
                        gtlcng = gtlcng[0]
                        if(gtl in pdldict.keys()):
                            if(len(pdldict[gtl])>0):
                                if(gtlcng in pdldict[gtl]):
                                    lemmacorr = 1
                                    predictedcng = gtlcng
                                    lemmaCNGcorr = 1
                                    pdldict[gtl].remove(gtlcng)
                                    with open('groundtruth2.csv','a') as fh:
                                        rwd = csv.writer(fh)
                                        row = [sentid,gtl,gtlcng,lemmacorr,lemmaCNGcorr,gtlcng,confcount]
                                        rwd.writerow(row)
                                else:
                                    lemmaround2.append(gtl)
                                    cnground2.append((gtlcng,confcount))
                            else:
                                with open('groundtruth2.csv','a') as fh:
                                        rwd = csv.writer(fh)
                                        row = [sentid,gtl,gtlcng,lemmacorr,lemmaCNGcorr,predictedcng,confcount]
                                        rwd.writerow(row)
                        else:
                             with open('groundtruth2.csv','a') as fh:
                                        rwd = csv.writer(fh)
                                        row = [sentid,gtl,gtlcng,lemmacorr,lemmaCNGcorr,predictedcng,confcount]
                                        rwd.writerow(row)
                # now all elements with lemmaCNGcorr ==1 are out of the way
                # reiterating for the lemmas which didnt have a cng but had a lemma earlier
                for i in range(len(lemmaround2)):
                    gtl = lemmaround2[i]
                    gtlcng = cnground2[i]
                    confcount = gtlcng[1]
                    gtlcng = gtlcng[0]
                    lemmacorr = 0
                    lemmaCNGcorr = 0
                    predictedcng = 'nil'
                    if(gtl in pdldict.keys()):
                        if(len(pdldict[gtl])>0):
                            lemmacorr = 1
                            predictedcng = pdldict[gtl][0]
                            pdldict[gtl].remove(pdldict[gtl][0])
                    with open('groundtruth2.csv','a') as fh:
                            rwd = csv.writer(fh)
                            row = [sentid,gtl,gtlcng,lemmacorr,lemmaCNGcorr,predictedcng,confcount]
                            rwd.writerow(row)
#                 print('done here')
            except Exception as e:
                print(e)
                print('been there')
#                 break
                continue
                