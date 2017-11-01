"""Convert pickles with protocol > 2 (4 in this file's pickle) to protocol 2"""

import pickle
files_list = ['featureSet_500samples_8L_0.p', 'featureSet_500samples_8L_1.p', 'featureSet_500samples_8L_2.p', 'featureSet_500samples_8L_3.p', 'featureSet_500samples_8L_4.p', 'featureSet_500samples_8L_5.p', 'featureSet_500samples_8L_6.p', 'featureSet_500samples_8L_7.p', 'featureSet_500samples_8L_8.p', 'featureSet_500samples_8L_9.p', 'featureSet_500samples_8L_10.p', 'featureSet_500samples_8L_11.p', 'featureSet_500samples_8L_12.p', 'featureSet_500samples_8L_13.p', 'featureSet_500samples_8L_14.p', 'featureSet_500samples_8L_15.p', 'featureSet_500samples_8L_16.p', 'featureSet_500samples_8L_17.p', 'featureSet_500samples_8L_18.p', 'featureSet_500samples_8L_19.p']

for ffile in files_list:
    print("Loading %s" % ffile)
    f = pickle.load(open(ffile, 'rb'), encoding='utf8')
    with open(('2_%s' % ffile), 'wb') as k:
        pickle.dump(f, k, 2)
    print("Done dumping")
