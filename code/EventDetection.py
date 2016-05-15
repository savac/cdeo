import utils.cat_parser as cat_parser
import utils.utils as utils
import copy
import subprocess
import os
import numpy as np
import itertools
import cdeo_config
reload(utils)

def getEventStanfordNLP(indoc):
    '''Finds the target entities specified by <targetEntityTextList> in the document <indoc>. 
    Returns new document with annotated target entities.''' 
    m_id = 0
    doc = copy.deepcopy(indoc)
    doc.Markables.EVENT_MENTION = [] # remove any previous event annotations
    #txt = utils.getRawText(doc).split(' ')
    #print doc.get_doc_id(), txt
    
    wanted_pos = ['VB','VBP', 'VBG', 'VBN', 'VBD', 'VBZ']
    n_sentences = len(doc.root[0][0])
    
    candidate_ids = []
    
    for s_ind in range(0, n_sentences):

        # check the POS tags
        for candidate in doc.root[0][0][s_ind][0]:
            pos = candidate[4].text
            if pos in wanted_pos:
                txt = candidate[0].text
                
                # TBD: this is dumb
                # now loop over all doc tokens and string match
                for tok in doc.token:
                    if tok.valueOf_ == txt:
                        tid = tok.get_t_id()
                        if tid not in [item for sublist in candidate_ids for item in sublist]:
                            candidate_ids += [[tid]]
                            
    for t in candidate_ids:
        # prep the token id list
        token_anchor_list = list()
        for t0 in t:
            tmp = cat_parser.token_anchorType1()
            tmp.set_t_id(t0)
            token_anchor_list.append(tmp)
        
        # build the EVENT_MENTION structure as in cat_parser
        event = cat_parser.EVENT_MENTIONType()
        event.set_token_anchor(token_anchor_list) 
        event.set_m_id(m_id)
        m_id+=1
        doc.Markables.EVENT_MENTION.append(event)
    return doc
