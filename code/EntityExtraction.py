import utils.cat_parser as cat_parser
import utils.utils as utils
import copy
import subprocess
import os
import lxml.etree as ET
import numpy as np
import itertools
import Levenshtein
import cdeo_config
reload(utils)


def getEntitiesRuleBased(indoc, targetEntityTextList):
    '''Finds the target entities specified by <targetEntityTextList> in the document <indoc>. 
    Returns new document with annotated target entities.''' 
    doc = copy.deepcopy(indoc)
    doc.Markables.ENTITY_MENTION = list()
    txt = utils.getRawText(doc).split(' ')
    #txt = 'Steve Jobs likes to talk about Steve . Jobs are rare . Jobs Steve are rarer .'.split(' ')
    
    m_id = 0
    for targetEntityText in targetEntityTextList:
        targetEntity = targetEntityText.split(' ')
        tList = list()
        targetEntityLength = len(targetEntity)
        for i in range(0, len(txt) - targetEntityLength):
            if txt[i:i+targetEntityLength] == targetEntity:
                tList.append(range(i, i+targetEntityLength))
        
        # build the ENTITY_MENTION list 
        for t in tList:
            # prep the token id list
            token_anchor_list = list()
            for t0 in t:
                tmp = cat_parser.token_anchorType1()
                tmp.set_t_id(t0 + 1) # NB: token t_id are +1 wrt their index
                token_anchor_list.append(tmp)
            
            # build the EVENT_MENTION structure as in cat_parser
            entity = cat_parser.ENTITY_MENTIONType()
            entity.set_token_anchor(token_anchor_list) 
            entity.set_m_id(m_id)
            m_id+=1
            entity.set_type(targetEntityText)
            doc.Markables.ENTITY_MENTION.append(entity)
    return doc

def getEntitiesStanfordNLP(indoc, targetEntityTextList):
    '''Finds the target entities specified by <targetEntityTextList> in the document <indoc>. 
    Returns new document with annotated target entities.''' 
    doc = copy.deepcopy(indoc)
    doc.Markables.ENTITY_MENTION = list()
    txt = utils.getRawTextLines(doc)
    #print doc.get_doc_id(), txt
    
    bufferDir = cdeo_config.getConfig("root_dir") + 'data/tmp/ner/'
    fname = bufferDir + str(doc.get_doc_id()) + '.txt.xml'
    if os.path.isfile(fname):
        f = open(fname, 'r')
        s = f.read()
        f.close()
        root = ET.fromstring(s)
    else:
        # TBD: hacky
        os.chdir(cdeo_config.getConfig("stanfordcorenlp_loc"))
        f = open('tmp_file.txt', 'w')
        f.write(txt.encode('ascii', 'ignore'))
        f.close()
        process = subprocess.Popen('java -cp stanford-corenlp-3.5.2.jar:stanford-corenlp-3.5.2-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-3.5.2.jar -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -ssplit.eolonly true -file tmp_file.txt', shell=True, stdout=subprocess.PIPE)
        process.wait()
        tree = ET.parse('tmp_file.txt.xml')
        root = tree.getroot()
        
        # save for later runs
        f = open('tmp_file.txt.xml', 'r')
        fin = f.read()
        f.close()
        f = open(fname, 'w')
        f.write(fin.decode('ascii', 'ignore')) # decode or encode
        f.close()
        os.chdir(cdeo_config.getConfig("root_dir") + 'code')
    
    txtListRaw = utils.getRawTextList(doc)
    txtList = [x.lower() for x in txtListRaw]
    m_id = 0
    for targetEntityText in targetEntityTextList:
        targetEntity = targetEntityText.lower().split(' ') # NB: converted to lower case
                
        # Rule-based entity extraction
        # Firstly get the entity matches in a 'moving window'- look at several consecutive tokens in text at once.
        # For example, if the target entity is made of 2 words, we look at blocks of 2+2 consecutive 
        # tokens and check if they contain the target entity i.e. we look at 4 tokens at once.
        #print targetEntity, '-------------'
        tList = list()
        targetEntityLength = len(targetEntity)
        extra_tokens = 1
        wanted = targetEntity
        for i in range(0, len(doc.token) - targetEntityLength - extra_tokens):
            given = txtList[i:i+targetEntityLength + extra_tokens]
            if set(wanted).issubset(given):
                ind_dict = dict((k,i) for i,k in enumerate(given))
                inter = set(given).intersection(wanted)
                indices = [ind_dict[x] for x in inter]
                tmp = [x+i+1 for x in indices]
                tList.append(tmp) # NB: the token ids start at 1
                #tList.append(range(i+1, i+targetEntityLength + extra_tokens+1)) # NB: the token ids start at 1
        
        # find the right coreference chain: strict mention match on the target entity
        wanted_coref = list()
        corefs = root[0][1]
        exit_flag = False
        for coref in corefs:
            # Look at the mentions inside this coreference. If any one striclty matches our target entity then
            # this is the correct coreference for this entity.
            for mention in coref:
                cpy = copy.deepcopy(mention)
                mention_text = mention[4].text.lower().split(' ')
                mention_text = utils.removePunctuationList(mention_text) # remove punctuation
                if targetEntity == mention_text:
                    #print targetEntity, mention_text
                    wanted_coref = coref
                    exit_flag = True
                    break
            if exit_flag:
                break
        
        # Secondly, try to match a partial text of the target entity. (We only look at
        # partial texts that is missing just one token from the original target entity text.)
        #if len(utils.getNumericTokens(targetEntity)) == 0:
        #if len(tList) == 0:
        permutations = list()
        extra_tokens = 0
        for i in range(0, len(targetEntity)):
            cpy = copy.copy(targetEntity)
            cpy.pop(i)
            if len(cpy) > 0:
                permutations.append(cpy)
        if len(targetEntity) > 2: # TBD: improve this
            for tok in targetEntity:
                permutations.append([tok])
                
        for i in range(0, len(doc.token) - targetEntityLength + 1 - extra_tokens): 
            for p in permutations:
                p_length = len(p)
                given = txtList[i:i+p_length+extra_tokens]
                if set(p).issubset(given): # TBD: use tokens from CAT files...
                    # just make sure we discriminate between company and product: Boeing vs Boeing 777
                    wanted_numeric_tokens = utils.getNumericTokens(targetEntity)
                    given_numeric_tokens = utils.getNumericTokens(given)
                    if set(wanted_numeric_tokens) == set(given_numeric_tokens):
                        #print targetEntity, p, txtList[i:i+p_length+extra_tokens]
                        tList.append(range(i+1, i+p_length+extra_tokens+1)) # NB: the token ids start at 1

        # find the right coreference chain: match on the partial target entity
        '''
        permutations = list()
        for i in range(0, len(targetEntity)):
            cpy = copy.copy(targetEntity)
            cpy.pop(i)
            if len(cpy) > 0:
                permutations.append(cpy)
        if len(wanted_coref) == 0:
            exit_flag = False
            for coref in corefs:
                # Look at the mentions inside this coreference. If any one striclty matches our partial target entity then
                # this is the correct coreference for this entity.
                for mention in coref:
                    cpy = copy.deepcopy(mention)
                    mention_text = cpy[4].text.lower().split(' ')
                    mention_text = utils.removePunctuationList(mention_text) # remove punctuation
                    for p in permutations:
                        if p == mention_text:
                            #print p, mention_text, coref, wanted_coref
                            wanted_coref = coref
                            exit_flag = True
                            break
                    if exit_flag:
                        break
                if exit_flag:
                    break
        #print wanted_coref, len(wanted_coref)
        '''
        
        levenshtein_threshold = cdeo_config.getConfig('levenshtein_threshold')
        
        # Thirdly, if we have been unsuccessful in finding a strict match, look at Levenshtein similarity of the strings
        lev_coref_search_list = [] #targetEntityText.lower()]
        if len(tList) == 0:
            if targetEntityLength > 1: # NB: use it for entities longer that 1 token
                for i in range(0, len(doc.token) - targetEntityLength):
                    given_tokens = txtList[i:i+targetEntityLength]
                    wanted = targetEntityText.lower()
                    given = ' '.join(given_tokens).encode('utf-8')
                    if Levenshtein.distance(wanted, given)/float(len(wanted)) <= levenshtein_threshold: # NB: this needs to be optimised        
                        # NB: the numeric parts of the target entity need to be matched exactly
                        # so we don't confuse a 777 with a 787
                        wanted_numeric_tokens = utils.getNumericTokens(targetEntity)
                        given_numeric_tokens = utils.getNumericTokens(given_tokens)
                        if set(wanted_numeric_tokens) == set(given_numeric_tokens):
                            #print 'Levenshtein: ', targetEntity, wanted, given
                            tList.append(range(i+1, i+targetEntityLength+1)) # NB: the token ids start at 1
                            lev_coref_search_list.append(given)
                        
            # find the coreference chain            
            exit_flag = False
            for coref in corefs:
                # Look at the mentions inside this coreference. If any one striclty matches our partial target entity then
                # this is the correct coreference for this entity.
                for mention in coref:
                    cpy = copy.deepcopy(mention)
                    mention_text = cpy[4].text.lower().split(' ')
                    mention_text = utils.removePunctuationList(mention_text) # remove punctuation
                    for p in lev_coref_search_list:
                        if p == mention_text:
                            #print p, mention_text, coref, wanted_coref
                            wanted_coref = coref
                            exit_flag = True
                            break
                    if exit_flag:
                        break
                if exit_flag:
                    break
                        
        #for lst in tList:
        #    s = ''
        #    for tok in lst:
        #        s += utils.getToken(doc, tok).valueOf_ + ' '
        #    print s, lst
            
        tListStrict = tList

        
        # Extract target entities from the coreference chain
        
        # find the number of sentences and their length
        senLenList = list()
        for sen in root[0][0]:
            senLenList.append(len(sen[0]))
        
        tList = list()
        docLen = len(doc.token)
        for mention in wanted_coref:
            # sentence
            sen = int(mention[0].text)
            offset = int(np.sum(senLenList[0:sen-1]))
            
            tstart = int(mention[1].text) + offset
            tstop = int(mention[2].text) + offset
            
            if tstop > docLen:
                tstop = docLen # trim this coref
            if tstart > docLen:
                continue # give up on this mention
            
            # TBD: try to remove noise words here
            
            tok = range(tstart, tstop) # NB: because the token ids start at 1
            if len(tok) <= len(targetEntity) + 2: # avoid very long coreferces
                #tList.append(list(tok))
                #print targetEntity, mention[4].text, tok
                
                # NB
                # Sometimes raw text and NER text tokens get missaligned
                # We need to make sure they line up!
                options_list = list()
                wanted = mention[4].text
                for i in range(0, len(doc.token) - len(tok)): # search for matches in the text
                    given = txtListRaw[i:i+len(tok)]
                    given = ' '.join(given)
                    if wanted == given:
                        options_list.append(range(i+1,i+len(tok)+1))
        
                #print options_list
                # find the best match
                best_match = 20 # this limits how far away we look for a match
                best_opt = []
                for opt in options_list:
                    diff = abs(tok[0] - opt[0])
                    if diff < best_match:
                        best_opt = opt
                        best_match = diff
                tok = best_opt # finally, adjust

                raw_text = []
                for t in tok:
                    raw_text.append(utils.getToken(doc, t).valueOf_)
                raw_text = ' '.join(raw_text)
                #print raw_text

                if mention[4].text == raw_text: # append to mention list only when aligned
                    tList.append(list(tok))
                    #print '@@@@',targetEntity, mention[4].text, tok
                

        #print tListStrict, tList

        # Combine the strict matches and the coreference list and remove duplicates
        tList = tList + tListStrict
        #tList = [list(x) for x in set(tuple(x) for x in tList)] # remove duplicates
        
        # Flatten and remove repeating mentions. Each token of every mention will be an entity mention on its own.
        tList = [item for sublist in tList for item in sublist]
        tList = list(set(tList))
        
        tListPruned = copy.deepcopy(tList)
        avoid_words_list = ["the", "'s", "a", "an", ",", ".", "that"]
        for mention in tList:
            thisText = utils.getToken(doc, mention).valueOf_.lower()
            if avoid_words_list.count(thisText) > 0:
                tListPruned.remove(mention)
        tList = tListPruned
            
        # finaly these mentions need to be in a list
        tList = [[x] for x in tList]

        #print '+++++++++++ Final: +++++++++++++'
        #for lst in tList:
        #    s = ''
        #    for tok in lst:
        #        s += utils.getToken(doc, tok).valueOf_ + ' '
        #    print s, lst

        # build the ENTITY_MENTION list 
        for t in tList:
            # prep the token id list
            token_anchor_list = list()
            for t0 in t:
                tmp = cat_parser.token_anchorType1()
                tmp.set_t_id(t0)
                token_anchor_list.append(tmp)
            
            # build the EVENT_MENTION structure as in cat_parser
            entity = cat_parser.ENTITY_MENTIONType()
            entity.set_token_anchor(token_anchor_list) 
            entity.set_m_id(m_id)
            m_id+=1
            entity.set_type(targetEntityText)
            doc.Markables.ENTITY_MENTION.append(entity)
        
        doc.root = root # attach the Stanford NLP NER parse to work out the features later
    return doc
