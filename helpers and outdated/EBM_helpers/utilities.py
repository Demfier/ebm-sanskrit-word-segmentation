import sys as Sys
import pickle, re
import numpy as np
from romtoslp import *

# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")

def pickleFixLoad(filename):
    return pickle.load(open(filename, 'rb'), encoding=u'utf-8')

def validatePickleName(fName):
    m = re.search("^[\w]*.p$", fName)
    if m != None:
        return(m.group(0))
    return("")

sandhiRules = pickle.load(open('sandhiRules.p','rb'))    
def CanCoExist_sandhi(p1, p2, name1, name2):
    # P1 must be less than P2
    # Just send it in the proper order
    if(p1 < p2):
        overlap = max((p1 + len(name1)) - p2, 0)
        if overlap == 0:
            return True
        if overlap == 1 or overlap == 2:
            # try:
            p1 = (name1[len(name1) - overlap:len(name1):], name2[0])
            p2 = (name1[-1], name2[0:overlap:])
            # print(name1, name2, p1, p2)
            # print(p1, p2)
            if p1 in sandhiRules:
                if(sandhiRules[p1]['length'] < len(p1[0]) + len(p1[1])):
                    return True
            if p2 in sandhiRules:
                if(sandhiRules[p2]['length'] < len(p2[0]) + len(p2[1])):
                    return True
            # except IndexError:
            #     print('Sandhi function Error: arguments were', (p1, p2, name1, name2))
            #     return False

    return False

def fix_w_new(word_new_obj):    
    # dicto= { 'asmad':'mad','yuzmad':'tvad','ayam':'idam','agn':'agni','ya':'yad','eza':'etad',
    #          'tad':'sa','vd':'vid','va':'vE','-tva':'tva','ptta':'pitta','mahat':'mahant','ndra':'indra',
    #          'ap':'api','h':'hi','t':'iti','tr':'tri','va':'iva'}

    dicto= { 'asmad':'mad','yuzmad':'tvad','ayam':'idam','agn':'agni','ya':'yad','eza':'etad',
             'vd':'vid','va':'vE','-tva':'tva','ptta':'pitta','mahat':'mahant','ndra':'indra',
             'ap':'api','h':'hi','t':'iti','tr':'tri','va':'iva'}

    for i in range(0,len(word_new_obj.lemmas)):
        word_new_obj.lemmas[i]= rom_slp(word_new_obj.lemmas[i])
        word_new_obj.lemmas[i]= word_new_obj.lemmas[i].split('_')[0]
        
        if word_new_obj.lemmas[i] in dicto:
            # print('CHANGED', word_new_obj.lemmas[i])
            word_new_obj.lemmas[i]= dicto[word_new_obj.lemmas[i]]
                
        if(word_new_obj.lemmas[i]== 'yad'):
            if word_new_obj.names== 'yadi':
                word_new_obj.lemmas[i]= 'yadi'
                
    return(word_new_obj)

def FixSentence(sentenceObj):
    for ci in range(len(sentenceObj.chunk)):
        for pos in sentenceObj.chunk[ci].chunk_words.keys():
            for wsi in range(len(sentenceObj.chunk[ci].chunk_words[pos])):
                sentenceObj.chunk[ci].chunk_words[pos][wsi] = fix_w_new(sentenceObj.chunk[ci].chunk_words[pos][wsi])
    
    return sentenceObj

def FillMissing(sentenceObj, dcsObj):
    for ci in range(len(sentenceObj.chunk)):
        corrLemmas = dcsObj.lemmas[ci]
        cli = 0
        iamdone = False
        for pos in sentenceObj.chunk[ci].chunk_words.keys():
            for wsi in range(len(sentenceObj.chunk[ci].chunk_words[pos])):
                ws = sentenceObj.chunk[ci].chunk_words[pos][wsi]
                for li in range(len(ws.lemmas)):
                    if ws.lemmas[li] == rom_slp(corrLemmas[cli]):
                        # print('MATCHED:', ws.lemmas[li], rom_slp(corrLemmas[cli]))
                        # print('CNG LIST:', ws.forms[li] if li < len(ws.forms) else [dcsObj.cng[ci][cli]])
                        
                        if li >= len(ws.forms):
                            a = ['']*(li + 1)
                            for i in range(len(ws.forms)):
                                a[i] = ws.forms[i]
                            a[li] = int(dcsObj.cng[ci][cli])
                            sentenceObj.chunk[ci].chunk_words[pos][wsi].forms = a

                        cli += 1
                        if cli == len(corrLemmas):
                            iamdone = True
                            break
                if iamdone:
                    break
            if iamdone:
                break
    return sentenceObj



# def loadSentence(fName, sntcPath):
#     try:
#         dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName)
#         sentenceObj = pickleFixLoad(sntcPath)
#         sentenceObj = FixSentence(sentenceObj)
#         sentenceObj = FillMissing(sentenceObj, dcsObj)
#     except (KeyError, EOFError, pickle.UnpicklingError) as e:
#         print('Failed to load', sntcPath)
#         return None, None
#     return(sentenceObj, dcsObj)

def loadSentence_with_rom_slp(fName, sntcPath):
    try:
        try:
            if fName[-1] == '2':
                dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName[:-1])
            else:
                dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName)
        except FileNotFoundError:
            dcsObj = None
        sentenceObj = pickleFixLoad(sntcPath)
        sentenceObj = FixSentence(sentenceObj)
    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        print('Failed to load', sntcPath)
        return None, None
    return(sentenceObj, dcsObj)

def loadSentence_nopre(fName, sntcPath):
    try:
        if fName[-1] == '2':
            dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName[:-1])
        else:
            dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName)
        sentenceObj = pickleFixLoad(sntcPath)
    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        print('Failed to load', sntcPath)
        return None, None
    return(sentenceObj, dcsObj)

preList = pickle.load(open('pvb.p', 'rb'))
def removePrefix(lemma):
    for pre in preList:
        m = re.match(pre, lemma)
        if(m != None):
            s = m.span()
            pat = lemma[s[0]:s[1]]
            return (lemma.split(pat)[1])
    return lemma

def GetSolutions(dcsObj):
    solution = [rom_slp(c) for arr in dcsObj.lemmas for c in arr]
    solution_no_pvb = [removePrefix(l) for l in solution]
    return (solution, solution_no_pvb)
    
def Accuracy(prediction, dcsObj):
    solution, solution_no_pvb = GetSolutions(dcsObj)
    # print('Solution:', solution)
    # print('Solution No Pvb:', solution_no_pvb)
    ac = 0
    for x in range(len(solution)):
        if(solution[x] in prediction):
            ac += 1
        # elif(solution_no_pvb[x] in prediction):
        #     ac += 1

    ac = 100*ac/len(solution)
    return ac

def FullCoverage(skt, dcs):
    # print('-'*40)
    # print('NEW FILE RCVD')
    goodFlag = True
    for ci in range(len(dcs.lemmas)):
        dlemmas = [rom_slp(l) for l in dcs.lemmas[ci]]
        slemmas = []
        chunk = skt.chunk[ci]
        for pos in chunk.chunk_words.keys():
            for wsi in range(len(chunk.chunk_words[pos])):
                ws = chunk.chunk_words[pos][wsi]
                [slemmas.append(wsl) for wsl in ws.lemmas]
        # print('DCS:', dlemmas)
        # print('SKT:', slemmas)
        for l in dlemmas:
            if l not in slemmas:
                # print(l, 'not found')
                goodFlag = False
                break
        if not goodFlag:
            break
#     print(goodFlag)
    return goodFlag

def GetFeatNameSet():
    mat_cngCount_1D = pickle.load(open('../NewData/gauravs/Temporary_1D/mat_cngCount_1D.p', 'rb'), encoding = u'utf-8')

    _full_cnglist = list(mat_cngCount_1D)
    _cg_count = len(mat_cngCount_1D)

    feats = {}
    fIndex = 0
    feats[fIndex] = ('L', 'L'); fIndex += 1;
    feats[fIndex] = ('L', 'C'); fIndex += 1;
    feats[fIndex] = ('L', 'T'); fIndex += 1;

    feats[fIndex] = ('C', 'L'); fIndex += 1;
    feats[fIndex] = ('C', 'C'); fIndex += 1;
    feats[fIndex] = ('C', 'T'); fIndex += 1;

    feats[fIndex] = ('T', 'L'); fIndex += 1;
    feats[fIndex] = ('T', 'C'); fIndex += 1;
    feats[fIndex] = ('T', 'T'); fIndex += 1;

    # Path Constraint - Length 2 - # _cg_count

    # LEMMA->CNG->LEMMA
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('L', cng_k, 'L')
    fIndex += _cg_count

    # LEMMA->CNG->CNG
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('L', cng_k, 'C')
    fIndex += _cg_count

    # LEMMA->CNG->TUP
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('L', cng_k, 'T')
    fIndex += _cg_count

    # CNG->CNG->LEMMA
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('C', cng_k, 'L')
    fIndex += _cg_count

    # CNG->CNG->CNG
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('C', cng_k, 'C')
    fIndex += _cg_count

    # CNG->CNG->TUP
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('C', cng_k, 'T')
    fIndex += _cg_count

    # TUP->CNG->LEMMA :: TOO MANY ZEROS
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('T', cng_k, 'L')
    fIndex += _cg_count

    # TUP->CNG->CNG :: TOO MANY ZEROS
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('T', cng_k, 'C')
    fIndex += _cg_count

    # TUP->CNG->TUP :: TOO MANY ZEROS
    for k in range(0, _cg_count):
        cng_k = _full_cnglist[k]
        feats[fIndex + k] = ('T', cng_k, 'T')
    fIndex += _cg_count

    # Path Constraint - Length 3 - # _cg_count^2

    # LEMMA->CGS->CGS->LEMMA
    for k1 in range(0, _cg_count):
        cng_k1 = _full_cnglist[k1]
        for k2 in range(0, _cg_count): 
            cng_k2 = _full_cnglist[k2]
            feats[fIndex + k1*_cg_count + k2] = ('L', cng_k1, cng_k2, 'L')
    fIndex += _cg_count**2

    # LEMMA->CGS->CGS->TUP
    for k1 in range(0, _cg_count):
        cng_k1 = _full_cnglist[k1]
        for k2 in range(0, _cg_count): 
            cng_k2 = _full_cnglist[k2]
            feats[fIndex + k1*_cg_count + k2] = ('L', cng_k1, cng_k2, 'T')
    fIndex += _cg_count**2

    # TUP->CGS->CGS->LEM
    for k1 in range(0, _cg_count):
        cng_k1 = _full_cnglist[k1]
        for k2 in range(0, _cg_count): 
            cng_k2 = _full_cnglist[k2]
            feats[fIndex + k1*_cg_count + k2] = ('T', cng_k1, cng_k2, 'L')
    fIndex += _cg_count**2

    # TUP->CGS->CGS->TUP
    for k1 in range(0, _cg_count):
        cng_k1 = _full_cnglist[k1]
        for k2 in range(0, _cg_count): 
            cng_k2 = _full_cnglist[k2]
            feats[fIndex + k1*_cg_count + k2] = ('T', cng_k1, cng_k2, 'T')
    fIndex += _cg_count**2
    return feats