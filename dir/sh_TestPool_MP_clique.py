import multiprocessing as mp
import TestPool_Unit
from shutil import copyfile
import numpy as np
import time
import sys
from optparse import OptionParser

from collections import defaultdict

def Evaluate(result_arr):
    print('Files Processed: ', len(result_arr))
    recalls = []
    recalls_of_word = []
    precisions = []
    precisions_of_words = []
    fully_Correct_l = 0
    fully_Correct_w = 0
    for entry in result_arr:
        (word_match, lemma_match, n_dcsWords, n_output_nodes) = entry
        recalls.append(lemma_match/n_dcsWords)
        recalls_of_word.append(word_match/n_dcsWords)

        precisions.append(lemma_match/n_output_nodes)
        precisions_of_words.append(word_match/n_output_nodes)
        if lemma_match == n_dcsWords:
            fully_Correct_l += 1
        if word_match == n_dcsWords:
            fully_Correct_w += 1
    print('Avg. Micro Recall of Lemmas: {}'.format(np.mean(np.array(recalls))))
    print('Avg. Micro Recall of Words: {}'.format(np.mean(np.array(recalls_of_word))))
    print('Avg. Micro Precision of Lemmas: {}'.format(np.mean(np.array(precisions))))
    print('Avg. Micro Precision of Words: {}'.format(np.mean(np.array(precisions_of_words))))
    rl = np.mean(np.array(recalls))
    pl = np.mean(np.array(precisions))
    print('F-Score of Lemmas: ', (2*pl*rl)/(pl+rl))
    print('Fully Correct Lemmawise: {}'.format(fully_Correct_l/len(recalls_of_word)))
    print('Fully Correct Wordwise: {}'.format(fully_Correct_w/len(recalls_of_word)))
    print('[{:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}]'.format(100*np.mean(np.array(recalls)), 100*np.mean(np.array(recalls_of_word)), 100*np.mean(np.array(precisions)), \
           100*np.mean(np.array(precisions_of_words)), 100*(2*pl*rl)/(pl+rl), 100*fully_Correct_l/len(recalls_of_word),\
           100*fully_Correct_w/len(recalls_of_word)))
    sys.stdout.flush()
    
tag = None
proc_count = 4

def main():
    global proc_count, tag
    ho_folders = {
        'PR2': 'skt_dcs_DS.bz2_4K_pmi_rfe_heldout',
        'BR2': 'skt_dcs_DS.bz2_4K_bigram_rfe_heldout',
        'PM2': 'skt_dcs_DS.bz2_4K_pmi_mir_heldout',
        'BM2': 'skt_dcs_DS.bz2_4K_bigram_mir_heldout',
        'PR3': 'skt_dcs_DS.bz2_1L_pmi_rfe_heldout',
        'BR3': 'skt_dcs_DS.bz2_1L_bigram_rfe_heldout',
        'PM3': 'skt_dcs_DS.bz2_1L_pmi_mir_heldout_again',
        'BM3': 'skt_dcs_DS.bz2_1L_bigram_heldout'
    }
    modelList = {
        'PR2': 'outputs/train_{}/nnet_e1_i400.p'.format('t2788294192566'),
        'BR2': 'outputs/train_{}/nnet_e1_i400.p'.format('t2789415023871'),
        'PM2': 'outputs/train_{}/nnet_e1_i400.p'.format('t2753954441900'),
        'BM2': 'outputs/train_{}/nnet_e1_i400.p'.format('t3401216067518'),
        'PR3': 'outputs/train_{}/nnet_e1_i400.p'.format('t2761370242287'),
        'BR3': 'outputs/train_{}/nnet_e1_i400.p'.format('t2779114903467'),
        'PM3': 'outputs/train_{}/nnet_e1_i400.p'.format('t2756013734745'),
        'BM3': 'outputs/train_{}/nnet_e1_i400.p'.format('t3471903174862')
    }
    modelFile = modelList[tag]
    print('Tag: {}, ModelFile: {}'.format(tag, modelFile))
    print('ProcCount: {}'.format(proc_count))
    _dump = True
    if _dump:
        _outFile = 'outputs/dump_predictions/{}_NLoss'.format(tag)
    else:
        _outFile = None
    print('OutFile: ', _outFile)

    # Backup the model file
    copyfile(modelFile, modelFile + '.bk')

    # Create Queue, Result array
    queue = mp.Queue()
    result_arr = []

    print('Source: ', '../NewData/{}/'.format(ho_folders[tag]))
    # Start 6 workers - 8 slows down the pc
    # proc_count = 4
    procs = [None]*proc_count
    for i in range(proc_count):
        vpid = i
        procs[i] = mp.Process(target = TestPool_Unit.pooled_Test, args = \
                              (modelFile, vpid, queue, '../NewData/{}/'.format(ho_folders[tag]), int(9600/proc_count), _dump, _outFile))
    # Start Processes
    for i in range(proc_count):
        procs[i].start()
        
    # Fetch partial results
    stillRunning = True
    printer_timer = 100
    while stillRunning:
        stillRunning = False
        for i in range(proc_count):
            p = procs[i]
            # print('Process with\t vpid: {}\t ->\t pid: {}\t ->\t running status: {}'.format(i, p.pid, p.is_alive()))
            if p.is_alive():
                stillRunning = True
        
        
        if printer_timer == 0:
            printer_timer = 100
            while not queue.empty():
                result_arr.append(queue.get())
            # Evaluate results till now
            if len(result_arr) > 0:
                Evaluate(result_arr)

        printer_timer -= 1
        
        time.sleep(1)
    while not queue.empty():
        result_arr.append(queue.get())
    Evaluate(result_arr)
    for i in range(proc_count):
        procs[i].join()
def setArgs(_tag, _pc):
    global proc_count, tag
    tag = _tag
    proc_count = _pc
    print('Tag, ProcCount: {}, {}'.format(tag, proc_count))
    
if __name__ == '__main__':
    

    #print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv))
    parser = OptionParser()
    parser.add_option("-t", "--tag", dest="tag",
                      help="Tag for feature set to use", metavar="TAG")
    parser.add_option("-p", "--procs", dest="proc_count", default = 4,
                      help="Number of child process", metavar="PROCS")

    (options, args) = parser.parse_args()

    options = vars(options)
    _tag = options['tag']
    if _tag is None:
        raise Exception('None is tag')
    pc = int(options['proc_count'])
    setArgs(_tag, pc)
    
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        