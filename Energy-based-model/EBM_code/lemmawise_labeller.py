import csv, os, pickle
import bz2
from optparse import OptionParser

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

def main(small_tag):
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
    bz_folder = '../NewData/{}/'.format(ho_folders[small_tag])
    files = []

    tag = '{}_NLoss_'.format(small_tag)
    outFile = 'outputs/dump_predictions/lemma_label_{}.csv'.format(small_tag)
    print('Writing to ', outFile)
    
    for f in os.listdir('outputs/dump_predictions/'):
        if tag in f:
            print('Adding ', f)
            files.append(f)

    with open(outFile, 'w') as out_fh:
        out_fh_csv = csv.writer(out_fh)
        fi = 0
        for root_file in files:
            with open(os.path.join('outputs/dump_predictions/', root_file)) as fh:
                print('Processing File: ', root_file)
                fh_csv = csv.reader(fh)
                for lr in fh_csv:
                    if fi % 100 == 0:
                        print('Files done: ', fi)
                    fi += 1
                    sent_id = lr[0]
                    dcs_name = sent_id + '.ds.bz2'
                    (nodelist_correct, _, _, nodelist_to_correct_mapping,\
                        _, _, _) = open_dsbz2(os.path.join(bz_folder, dcs_name))
                    for rx in range(5):
                        lr = next(fh_csv)[1:]
                        if rx == 3:
                            iam = [int(x) for x in lr]
                    for i in range(len(nodelist_correct)):
                        out_fh_csv.writerow([sent_id, nodelist_correct[i].lemma, 1*(nodelist_to_correct_mapping[i] in iam)])

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--tag", dest="tag",
                      help="Tag for feature set to use", metavar="TAG")

    (options, args) = parser.parse_args()

    options = vars(options)
    _tag = options['tag']
    if _tag is None:
        raise Exception('None is tag')
    print(_tag)
    main(_tag)