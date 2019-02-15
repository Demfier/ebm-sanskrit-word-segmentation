#Loading of SKT Pickles
from romtoslp import rom_slp
from json import *
import pprint
from utilities import *
class word_new():
    def __init__(self,names):
        self.lemmas=[]
        self.names=names
        self.urls=[]
        self.forms=[]

class chunks:
    def __init__(self,chunk_name):
        self.chunk_name=chunk_name
        self.chunk_words={}

class sentences:
    def __init__(self,sent_id,sentence):
        self.sent_id=sent_id
        self.sentence=sentence
        self.chunk=[]

# def getCNGs(formsDict):
#         l = []
#         if type(formsDict) == int or type(formsDict) == str:
#             return [int(formsDict)]
#         else:
#             for form, configs in formsDict.items():
#                 for c in configs:
#                     if(form == 'verbform'):                
#                         continue
#                     else:
#                         l.append(wtc_recursive(form, configs))
#             return list(set(l))

class SentenceError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(SentenceError, self).__init__(message)

def SeeSentence(sentenceObj):
    print('SKT ANALYZE')
    print('-'*15)
    print(sentenceObj.sentence)
    zz = 0
    # (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(sentenceObj)
    # for cid in chunkDict.keys():
    #     print('Analyzing:', rom_slp(sentenceObj.chunk[cid].chunk_name))
    #     for pos in chunkDict[cid].keys():
    #         tupIds = chunkDict[cid][pos]
    #         for ti in tupIds:
    #             print('%d :' % (pos, ), end = ' ')
    #             print(tuplesMain[ti][0][1], end=' ')
    #             for tup in tuplesMain[ti]:
    #                 print([zz, tup[2], tup[3]], end=' ')
    #                 zz += 1
    #             print('')
    #     print('-'*25)

    for chunk in sentenceObj.chunk:
        print("Analyzing ", rom_slp(chunk.chunk_name))
        for pos in chunk.chunk_words.keys():
            for word_sense in chunk.chunk_words[pos]:
                word_sense = fix_w_new(word_sense)
                print(pos, ": ", rom_slp(word_sense.names), word_sense.lemmas, word_sense.forms)
                # for formsDict in word_sense.forms:
                #     print(getCNGs(formsDict))
    print()

def getWord(sentenceObj, cid, pos,kii):
    ch = sentenceObj.chunk[cid]
    word = ch.chunk_words[pos][kii]
    return {'lemmas': word.lemmas, 'forms':word.forms, 'names':word.names}

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
from wordTypeCheckFunction import *
import pickle

"""
SentencePreprocess:
-------------------
    Read a sentence obj and create + return the following objects 

    -> chunkDict: chunk_id -> position -> index in lemmaList (nested dictionary)
    -> lemmaList: list of possible words as a result of word segmentation
    -> revMap2Chunk: Map word in wordList to (cid, position) in chunkDict
    -> qu: Possible query nodes
"""
v2t = pickle.load(open('verbs_vs_cngs_matrix_countonly.p', 'rb'), encoding=u'utf8')
def wtc_recursive(form, c):
    if type(c) ==list:
        for cc in c:
            return wtc_recursive(form, cc)
    else:
        return wordTypeCheck(form, c)

def CanBeQuery(chunk):
    allLemmas = []
    for pos, words in chunk.chunk_words.items():
        for word in words:
            for lemma in word.lemmas:
                if lemma != '':
                    allLemmas.append(lemma)
    if(len(allLemmas) == 1):
        return True

def Get_QCs(tuplesMain, chunkDict):
    # Form NON-competitor dictionary - Query - Candidate Pairs
    qc_pairs = {}
    nodeList = [t for ts in tuplesMain for t in ts]
    
    for ni in range(len(nodeList)):
        qc_pairs[ni] = set(range(len(nodeList))) - set([ni])

    for cid in chunkDict.keys():
        # Neighbours
        for pos1 in chunkDict[cid].keys():
            for pos2 in chunkDict[cid].keys():
                if pos1 <= pos2:
                    nList1 = []
                    for ti1 in chunkDict[cid][pos1]:
                        for tup1 in tuplesMain[ti1]:
                            nList1.append(tup1[0])
                    nList2 = []
                    for ti2 in chunkDict[cid][pos2]:
                        for tup2 in tuplesMain[ti2]:
                            nList2.append(tup2[0])
                    nList1 = set(nList1)
                    nList2 = set(nList2)
                    for n1 in nList1:
                        qc_pairs[n1] = qc_pairs[n1] - nList1

                    for n2 in nList2:
                        qc_pairs[n2] = qc_pairs[n2] - nList2

                    if pos1 < pos2:
                        for n1 in nList1:
                            for n2 in nList2:
                                if not CanCoExist_sandhi(pos1, pos2, nodeList[n1][1], nodeList[n2][1]):
                                    qc_pairs[n1] = qc_pairs[n1] - set([n2])
                                    qc_pairs[n2] = qc_pairs[n2] - set([n1])
                                    
    return qc_pairs

'''
===================
SentencePreprocess
===================
forceQuery: Setting it true will make the longest word available a query if no
            other query is available
'''
def SentencePreprocess(sentenceObj, forceQuery = False):
    """
    Considering word names only
    ***{Word forms or cngs can also be used}
    """
    def getCNGs(formsDict):
        if type(formsDict) == int or type(formsDict) == str:
            return [int(formsDict)]
        else:
            l = []
            for form, configs in formsDict.items():
                for c in configs:
                    if(form == 'verbform'):
                        continue
                    else:
                        l.append(wtc_recursive(form, c))
            return list(set(l))

    chunkDict = {}
    lemmaList = []
    wordList = []
    cngList = []
    revMap2Chunk = []
    qu = []
    tuplesMain = []

    cid = -1
    tidExclusive = 0

    ## Traverse sentence and form data-structures
    for chunk in sentenceObj.chunk:
        # print(chunk.chunk_name)
        cid = cid+1
        chunkDict[cid] = {}
        for pos in chunk.chunk_words.keys():
            tupleSet = {}
            chunkDict[cid][pos] = []
            for word_sense in chunk.chunk_words[pos]:
                # word_sense = fix_w_new(word_sense)
                nama = rom_slp(word_sense.names)
                if nama == '':
                    raise SentenceError('Empty Name Detected')
                if(len(word_sense.lemmas) > 0 and len(word_sense.forms) > 0):
                    tuples = []
                    for lemmaI in range(len(word_sense.lemmas)):
                        # lemma = rom_slp(word_sense.lemmas[lemmaI].split('_')[0]) # NOT REQUIRED - DONE IN FIX_W_NEW
                        lemma = word_sense.lemmas[lemmaI]
                        if lemma == '':
                            continue
                        tempCNGs = getCNGs(word_sense.forms[lemmaI])
                        for cng in tempCNGs:
                            # UPDATE LISTS
                            newT_Key = (lemma, cng)
                            newT = (tidExclusive, nama, lemma, cng)
                            if(newT_Key not in tupleSet):
                                tupleSet[newT_Key] = 1
                                tuples.append(newT) # Remember the order
                                lemmaList.append(lemma)
                                wordList.append(nama)
                                cngList.append(cng)
                                revMap2Chunk.append((cid, pos, len(tuplesMain)))
                                tidExclusive += 1

                    if(len(tuples) > 0):
                        # print(tuples)
                        k = len(tuplesMain)
                        chunkDict[cid][pos].append(k)
                        tuplesMain.append(tuples)

    ## Find QUERY nodes now
    for cid in chunkDict.keys():
        tuples = []
        for pos in chunkDict[cid].keys():
            tupIds = chunkDict[cid][pos]
            for tupId in tupIds:
                [tuples.append((pos, tup[0], tup[1])) for tup in tuplesMain[tupId]]
        for u in range(len(tuples)):
            tup1 = tuples[u]
            quFlag = True
            for v in range(len(tuples)):
                if(u == v):
                    continue
                tup2 = tuples[v]
                
                # '''
                # FIXME: REMOVE TRY CATCH
                # '''
                # try:
                if(tup1[0] < tup2[0]):
                    if not CanCoExist_sandhi(tup1[0], tup2[0], tup1[2], tup2[2]):
                        ## Found a competing node - hence can't be a query
                        quFlag = False
                        break
                elif(tup1[0] > tup2[0]):
                    if not CanCoExist_sandhi(tup2[0], tup1[0], tup2[2], tup1[2]):
                        ## Found a competing node - hence can't be a query
                        quFlag = False
                        break
                else:
                    quFlag = False
                    break

                # except IndexError:
                #     print('From SentencePreprocess IndexError:', sentenceObj.sent_id)
                #     raise IndexError

            if quFlag:
                qu.append(tup1[1])

    # if len(qu) == 0:
    #     print('No query available')
    #     maxI = 0
    #     for i in range(len(wordList)):
    #         if len(wordList[i]) > len(wordList[maxI]):
    #             maxI = i
    #         elif len(wordList[i]) == len(wordList[maxI]):
    #             # Check the competitor count

    #     print(wordList[maxI], 'is forced query')

    verbs = []
    i = -1
    for w in lemmaList:
        i += 1
        if w in list(v2t.keys()):
            verbs.append(i)


    # pprint.pprint(tuplesMain)
    # pprint.pprint(chunkDict)
    # pprint.pprint(revMap2Chunk)
    
    qc_pairs = Get_QCs(tuplesMain, chunkDict)
    
    '''
    qu = [] # Have to remove it later
    '''
    # print(chunkDict)
    if len(qu) == 0 and len(lemmaList) > 0:
        lens = np.array([len(t[1]) for ts in tuplesMain for t in ts])
        cw = [(t[0], t[1]) for ts in tuplesMain for t in ts]
        round1 = np.where(lens == np.max(lens))[0]
        hits = [len(qc_pairs[r]) for r in round1]
        finalist = round1[np.where(hits == np.min(hits))][0]
        qu.append(finalist)
    
    return (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain, qc_pairs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
