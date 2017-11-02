from multiprocessing import Process
import multiprocessing as mp
import os, sys
from sentences import *
import numpy as np
from Train_n_Save_NNet import *

def pooled_Test(modelFile, vpid, queue, testfolder, filePerProcess = 100, _dump = False, _outFile = None):
    n_chkpt = 100
    print('Child process with vpid:{}, pid:{} started.'.format(vpid, os.getpid()))
    trainer = Trainer()
    trainer.Load(modelFile)

    TestFiles = []
    for f in os.listdir(testfolder):
        if '.ds.bz2' in f:
            TestFiles.append(f)
            
    print('vpid:{}: Range is {} -> {} / {}'.format(vpid, vpid*filePerProcess, vpid*filePerProcess + filePerProcess, len(TestFiles)))
    if _dump:
        _outFile = '{}_proc{}.csv'.format(_outFile, vpid)
        with open(_outFile, 'w') as fh:
            print('File refreshed', _outFile)
            
    loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT_ho.p', 'rb'))
    loaded_DCS = pickle.load(open('../Simultaneous_DCS_ho.p', 'rb'))
    
    #loader = pickle.load(open('../bz2Dataset_10K.p', 'rb'))
    #TestFiles = loader['TestFiles']
    #TrainFiles = loader['TrainFiles']

    for i in range(vpid*filePerProcess, vpid*filePerProcess + filePerProcess):
        #if i % n_chkpt == 0:
            #print('Checkpoint {}, vpid: {}'.format(i, vpid))
            #sys.stdout.flush()
        fn = TestFiles[i]
        fn = fn.replace('.ds.bz2', '.p2')

        dsbz2_name = testfolder + TestFiles[i]
        
        sentenceObj = loaded_SKT[fn]
        dcsObj = loaded_DCS[fn]
        try:
            if _dump:
                results = trainer.Test(sentenceObj, dcsObj, dsbz2_name, _dump=True, _outFile = _outFile)
            else:
                results = trainer.Test(sentenceObj, dcsObj, dsbz2_name)
        except EOFError as e:
            print('BADFILE', dsbz2_name)

        if results is not None:
                queue.put(results)
    print('Child process with vpid:{}, pid:{} closed.'.format(vpid, os.getpid()))

