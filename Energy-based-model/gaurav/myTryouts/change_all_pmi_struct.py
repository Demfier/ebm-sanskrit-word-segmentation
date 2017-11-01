import pickle

all_pmi = pickle.load(open('all_pmi.p', 'rb'), encoding='utf8')

all_pmi_Dict = {}

for entry in all_pmi:
    all_pmi_Dict[('%s^%s' % (entry[0], entry[1]))] = entry[2]

with open('all_pmi_dict.p', 'wb') as f:
    pickle.dump(all_pmi_Dict, f)
