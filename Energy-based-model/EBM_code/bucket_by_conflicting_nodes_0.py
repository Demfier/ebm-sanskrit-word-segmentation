import pickle
import os
import bz2


def harmonic(P, R):
    """For Calculation of F-Score since it is the HM of P and R"""
    return(2 * P * R / float(P + R))
# Test on a couple of files
base_path_csv = '/home/rs/15CS91R05/gaurav/myTryouts/init_results/prediction_csvs/'
base_path_bz2 = '/home/rs/15CS91R05/Bishal/NewData/skt_dcs_DS.bz2_1L_bigram_heldout_dev/'

pred_csvs = os.listdir(base_path_csv)


"""Task 5: See data from number of conflicts
    Approach: Select a node from DCS, take count of conflicting nodes using the conflictsDict_correct
"""

# Function to open bz2 files (that contains both DCS & SKT info)


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

bucket_by_conflicting_nodes = {}
num_conflicting_nodes = set()
csv = open(base_path_csv + pred_csvs[0], 'r').readlines()

for line in range(0, len(csv), 6):
    head_line = csv[line].strip().split(',')
    fname = head_line[0]
    print("Bz2 File number", fname, line / 6)

    (nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
        nodelist, conflicts_Dict, featVMat) = open_dsbz2(base_path_bz2 + fname + '.ds.bz2')

    assert len(nodelist_correct) == len(conflicts_Dict_correct)
    for node in conflicts_Dict_correct:
        lemma = nodelist_correct[node].lemma
        conflicting_nodes_count = len(conflicts_Dict_correct[node])
        if conflicting_nodes_count not in num_conflicting_nodes:
            num_conflicting_nodes.add(conflicting_nodes_count)
            bucket_by_conflicting_nodes[conflicting_nodes_count] = {'lemmas': set(), 'precision': [0, 0], 'recall': [0, 0]}

        bucket_by_conflicting_nodes[conflicting_nodes_count]['lemmas'].add(lemma)

    data = csv[line + 5].strip().split(',')
    word_recall = float(data[1]) / float(data[3])
    lemma_recall = float(data[2]) / float(data[3])
    word_precision = float(data[1]) / float(data[4])
    lemma_precision = float(data[2]) / float(data[4])

    bucket_by_conflicting_nodes[conflicting_nodes_count]['recall'][0] += word_recall
    bucket_by_conflicting_nodes[conflicting_nodes_count]['recall'][1] += lemma_recall
    bucket_by_conflicting_nodes[conflicting_nodes_count]['precision'][0] += word_precision
    bucket_by_conflicting_nodes[conflicting_nodes_count]['precision'][1] += lemma_precision

# for conflicting_count in bucket_by_conflicting_nodes:

    # # Average P & R
    # bucket_by_conflicting_nodes[conflicting_count]['precision'][0] /= bucket_by_conflicting_nodes[conflicting_count]['num_lemmas']
    # bucket_by_conflicting_nodes[conflicting_count]['precision'][1] /= bucket_by_conflicting_nodes[conflicting_count]['num_lemmas']
    # bucket_by_conflicting_nodes[conflicting_count]['recall'][0] /= bucket_by_conflicting_nodes[conflicting_count]['num_lemmas']
    # bucket_by_conflicting_nodes[conflicting_count]['recall'][1] /= bucket_by_conflicting_nodes[conflicting_count]['num_lemmas']
    #
    # # Find F-Score
    # wrd_fscore = harmonic(bucket_by_conflicting_nodes[conflicting_count]['precision'][0], bucket_by_conflicting_nodes[conflicting_count]['recall'][0])
    # lma_fscore = harmonic(bucket_by_conflicting_nodes[conflicting_count]['precision'][1], bucket_by_conflicting_nodes[conflicting_count]['recall'][1])
    # bucket_by_conflicting_nodes[conflicting_count]['fscore'] = [wrd_fscore, lma_fscore]

with open('final_task_gaurav/bucket_by_conflicting_nodes_0.p', 'wb') as f:
    pickle.dump(bucket_by_conflicting_nodes, f)
