def splitTillPeriod(config,listInput): #see that config is not empty and is of type string
    #returns config sans first part and firstpart is appended to listInput
    
    configList=list(config)
    out=''
    periodIndex=0
    val=''
    for i,val in enumerate(configList):
       a=2
       periodIndex=i
       if val=='.':
        break
       if val!=" ": 
           out=out+val; 
    if val!=".":
        config1=" ".join(config.split())
        listInput.append(config1)
        return ""
    else:    
        config1="".join(configList[(periodIndex+1):])
        listInput.append(out)
        return config1

def wordTypeCheck(form,config):
    #if it is noun Im assuming it has 3 parts
    #form is noun or verb or...

    # print(form, config)
    
    nounMapping={28:	'xt?',  29:	'Nom. sg. masc.',  30:	'Nom. sg. fem.',  31:	'Nom. sg. neutr.',  32:	'Nom. sg. adj.',  33:	'xt?',  34:	'Nom. du. masc.',  35:	'Nom. du. fem.',  36:	'Nom. du. neutr.',  37:	'Nom. du. adj.',  38:	'xt?',  39:	'Nom. pl. masc.',  40:	'Nom. pl. fem.',  41:	'Nom. pl. neutr.',  42:	'Nom. pl. adj.',  48:	'xt?',  49:	'Voc. sg. masc.',  50:	'Voc. sg. fem.',  51:	'Voc. sg. neutr.',  54:	'Voc. du. masc.',  55:	'Voc. du. fem.',  56:	'Voc. du. neutr.',  58:	'xt?',  59:	'Voc. pl. masc.',  60:	'Voc. pl. fem.',  61:	'Voc. pl. neutr.',  68:	'xt?',  69:	'Acc. sg. masc.',  70:	'Acc. sg. fem.',  71:	'Acc. sg. neutr.',  72:	'Acc. sg. adj.',  73:	'xt?',  74:	'Acc. du. masc.',  75:	'Acc. du. fem.',  76:	'Acc. du. neutr.',  77:	'Acc. du. adj.',  78:	'xt?',  79:	'Acc. pl. masc.',  80:	'Acc. pl. fem.',  81:	'Acc. pl. neutr.',  82:	'Acc. pl. adj.',  88:	'xt?',  89:	'Instr. sg. masc.',  90:	'Instr. sg. fem.',  91:	'Instr. sg. neutr.',  92:	'Instr. sg. adj.',  93:	'xt?',  94:	'Instr. du. masc.',  95:	'Instr. du. fem.',  96:	'Instr. du. neutr.',  97:	'Instr. du. adj.',  98:	'xt?',  99:	'Instr. pl. masc.',  100:	'Instr. pl. fem.',  101:	'Instr. pl. neutr.',  102:	'Instr. pl. adj.',  108:	'xt?',  109:	'Dat. sg. masc.',  110:	'Dat. sg. fem.',  111:	'Dat. sg. neutr.',  112:	'Dat. sg. adj.',  114:	'Dat. du. masc.',  115:	'Dat. du. fem.',  116:	'Dat. du. neutr.',  117:	'Dat. du. adj.',  118:	'xt?',  119:	'Dat. pl. masc.',  120:	'Dat. pl. fem.',  121:	'Dat. pl. neutr.',  122:	'Dat. pl. adj.',  128:	'xt?',  129:	'Abl. sg. masc.',  130:	'Abl. sg. fem.',  131:	'Abl. sg. neutr.',  132:	'Abl. sg. adj.',  134:	'Abl. du. masc.',  135:	'Abl. du. fem.',  136:	'Abl. du. neutr.',  137:	'Abl. du. adj.',  138:	'xt?',  139:	'Abl. pl. masc.',  140:	'Abl. pl. fem.',  141:	'Abl. pl. neutr.',  142:	'Abl. pl. adj.',  148:	'xt?',  149:	'Gen. sg. masc.',  150:	'Gen. sg. fem.',  151:	'Gen. sg. neutr.',  152:	'Gen. sg. adj.',  153:	'xt?',  154:	'Gen. du. masc.',  155:	'Gen. du. fem.',  156:	'Gen. du. neutr.',  157:	'Gen. du. adj.',  158:	'xt?',  159:	'Gen. pl. masc.',  160:	'Gen. pl. fem.',  161:	'Gen. pl. neutr.',  162:	'Gen. pl. adj.',  168:	'xt?',  169:	'Loc. sg. masc.',  170:	'Loc. sg. fem.',  171:	'Loc. sg. neutr.',  172:	'Loc. sg. adj.',  173:	'xt?',  174:	'Loc. du. masc.',  175:	'Loc. du. fem.',  176:	'Loc. du. neutr.',  177:	'Loc. du. adj.',  178:	'xt?',  179:	'Loc. pl. masc.',  180:	'Loc. pl. fem.',  181:	'Loc. pl. neutr.',  182:	'Loc. pl. adj.',  }
    verbMapping1={1: 'pr. [*] ac.', 2: 'opt. [*] ac.', 3: 'imp. [*] ac', 4: 'impft. [*] ac.', 5: 'fut. ac/ps.', 6: 'cond. ac/ps.', 7: 'per. fut. ac/ps.', 8: 'aor. [1] ac/ps.', 9: 'aor. [2] ac/ps.', 10: 'aor. [3] ac/ps.', 11: 'aor. [4] ac/ps.', 12: 'aor. [5] ac/ps.', 13: 'aor. [7] ac/ps.', 14: 'ben. ac/ps.', 15: 'pft. ac.', 16: 'per. pft.', 19: 'pp.', 20: 'ppa.', 21: 'pfp.', 22: 'inf.', 23: 'abs.', 24: 'pr. ps.', 26: 'imp. ps.', 27: 'impft. ps.', 28: 'aor. ps.', 29: 'opt. ps.', 30: 'ou', }
    verbMapping2={1: 'sg. 1', 2: 'sg. 2', 3: 'sg. 3', 4: 'du. 1', 5: 'du. 2', 6: 'du. 3', 7: 'pl. 1', 8: 'pl. 2', 9: 'pl. 3', }



    if form=='indeclinable':
        if config=='part.':
            return 2
        elif config=='conj.':
            return 2
        elif config=='abs.':
            return -230
        elif config=='prep.':
            return 2
        elif config=='ind.':
            return 2
        elif config=='ca. abs.':
            return -230
        else:
            return 'config is invalid'
        
    elif form=='compound':
        if config=='iic.':
            return 3
        elif config=='iiv.':
            return 3
        else:
            return 'config is invalid'
        
    elif form=='undetermined':
        if config=='adv.':
            return 2
        elif config=='und.':
            return 1
        elif config=='tasil':
            return 1
        else:
            return 'config is invalid'
    
    elif form=='noun':
#         print("entered noun")
        config1=config
        x=[]
        config1=splitTillPeriod(config1,x)
        one=x[0]
        x=[]
        config1=splitTillPeriod(config1,x)
        two=x[0]
        x=[]
        config1=splitTillPeriod(config1,x)
        three=x[0]
        
        isAdj=0
        if three=='*':
            three='n'
            isAdj=1
            
        for i in nounMapping.keys():
            if one!='i'and one!='g':
                if one[len(one)-2:] in nounMapping[i]:
                    if two in nounMapping[i]:
                    
                        if three in nounMapping[i]:
                            if(isAdj==0):
                              return i  
                            else:
                                return i+1
                                      
               
            elif one=='i':
                 if 'Instr' in nounMapping[i]:
                    if two in nounMapping[i]:
                                       
                      if three=='n':
                        if 'neutr' in nounMapping[i]:
                             if(isAdj==0):
                              return i  
                             else:
                                return i+1
                    
                      elif three in nounMapping[i]:
                           return i 
            elif one=='g':
                if 'Gen' in nounMapping[i]:
                   if two in nounMapping[i]:
                                       
                      if three=='n':
                        if 'neutr' in nounMapping[i]:
                             if(isAdj==0):
                              return i  
                             else:
                                return i+1
                    
                      elif three in nounMapping[i]:
                                return i 
        
    elif form=='verb':
        unit=0
        ten=0
        #to remove ca des int
        x=[]
        configActual=config
        config=splitTillPeriod(config,x)
        if(x[0]=='ca' or x[0]=='des' or x[0]=='int'):
            y=2 #do nothing
        else:
            config=configActual
        #if [vn.] is present
        if 'vn.' in config:
            config=config.replace('vn.','')       
        
        x=[]
        config=splitTillPeriod(config,x)
        
        one=x[0]
        two=''
        three=''
        ONE=''
        TWO=''
        
        if config!='':
            x=[]
            config=splitTillPeriod(config,x)
            temp=x[0]
            if temp!='sg'and temp!='pl' and temp!='du':
                two=temp
            else:
                ONE=temp
         
        if config!='':
            x=[]
            config=splitTillPeriod(config,x)
            temp=x[0]
            print
            if temp!='sg'and temp!='pl' and temp!='du':
                if ONE=='':
                    three=temp
            elif ONE!='':
                TWO=temp
            else:
                ONE=temp
        if config!='':
            x=[]
            config=splitTillPeriod(config,x)
            temp=x[0]
            if temp=='sg'or temp=='pl' or temp=='du':
                ONE=temp
            elif temp=='1'or temp=='2' or temp=='3':
                TWO=temp
        
        if config!='':
            x=[]
            config=splitTillPeriod(config,x)
            temp=x[0]           
            if temp=='1'or temp=='2' or temp=='3':
                TWO=temp  
        
        for i in verbMapping2.keys():
            if ONE!='':
                if ONE in verbMapping2[i] and TWO in verbMapping2[i]:
                   unit=i
                   break
                
        if one=='pp':
            ten=19
        if one=='ppa':
            ten=20
        if one=='pfp':
            ten=21 
        if one=='inf':
            ten=22
        if one=='abs':
            ten=23
        if one=='inj':
            ten=30
            
        if one=='pr' or one=='ppr':
            if two=='ps':
                ten=24
        if one=='imp':
            if two=='ps':
                ten=26
        if one=='impft':
            if two=='ps':
                ten=27
        if one=='aor':
            if two=='ps':
                ten=28
        if one=='opt':
            if two=='ps':
                ten=29  
                
        if one=='pr'or one=='ppr':
            if 'ac' in two or 'md' in two:
                ten=1
        if one=='opt':
            if 'ac' in two or 'md' in two:
                ten=2        
        if one=='imp':
            if 'ac' in two or 'md' in two:
                ten=3
        if one=='impft':
            if 'ac' in two or 'md' in two:
                ten=4
        if one=='pft' or one=='ppf':
            if 'ac' in two or 'md' in two:
                ten=15
             
        if one=='per':
            if two=='pft':
                ten=16
                
        
        if one=='fut' or one=='pfu':
            if 'ac' in two or 'ps' in two or 'md' in two:
                ten=5
        if one=='cond':
            if 'ac' in two or 'ps' in two or 'md' in two:
                ten=6
        if one=='ben':
            if 'ac' in two or 'ps' in two or 'md' in two:
                ten=14        
        
        if one=='aor':
            if 'ac' in two or 'ps' in two or 'md' in two:
                if '1' in two:
                    ten=8
                if '2' in two:
                    ten=9
                if '3' in two:
                    ten=10 
                if '4' in two:
                    ten=11
                if '5' in two or '6' in two:
                    ten=12
                if '7' in two:
                    ten=13    
        
        if one=='per':
            if two=='fut':
                if (('ac' in three) or ('ps' in three) or 'md' in three):
                    ten=7
                    
        if ten!=0:
            output=-1*(ten*10+unit)
            return output
        else:
            x=3
        
    else:
        return 'none'
