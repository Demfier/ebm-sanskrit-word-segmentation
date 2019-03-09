import multiprocessing as mp
import TestPool_Unit
from shutil import copyfile

def Evaluate(result_arr):
    print('Files Processed: ', len(result_arr))
    recalls = []
    recalls_of_word = []
    precisions = []
    precisions_of_words = []
    for entry in result_arr:
        (word_match, lemma_match, n_dcsWords, n_output_nodes) = entry
        recalls.append(lemma_match/n_dcsWords)
        recalls_of_word.append(word_match/n_dcsWords)

        precisions.append(lemma_match/n_output_nodes)
        precisions_of_words.append(word_match/n_output_nodes)
    print('Avg. Micro Recall of Lemmas: {}'.format(np.mean(np.array(recalls))))
    print('Avg. Micro Recall of Words: {}'.format(np.mean(np.array(recalls_of_word))))
    print('Avg. Micro Precision of Lemmas: {}'.format(np.mean(np.array(precisions))))
    print('Avg. Micro Precision of Words: {}'.format(np.mean(np.array(precisions_of_words))))
    
modelFile = 'outputs/train_nnet_t764815831413.p'
# Backup the model file
copyfile(modelFile, modelFile + '.bk')

modelFile = modelFile + '.bk'

# Create Queue, Result array
queue = mp.Queue()
result_arr = []

# Start 6 workers - 8 slows down the pc
proc_count = 10
procs = [None]*proc_count
for i in range(proc_count):
    vpid = i
    procs[i] = mp.Process(target = TestPool_Unit.pooled_Test, args = (modelFile, vpid, queue, 700))

# Start Processes
for i in range(proc_count):
    procs[i].start()

# Properly Join
for i in range(proc_count):
    procs[i].join()
    
# Fetch partial results
while not queue.empty():
    result_arr.append(queue.get())

# Evaluate results till now
Evaluate(result_arr)
