import utils.cat_parser as cat_parser
import utils.utils as utils
import copy
import requests
import json
import lxml.etree as ET
import os.path

def getTimestamps(indoc):
    doc = copy.deepcopy(indoc)
    #doc.Markables.TIMEX3 = list()
    
    bufferDir = '../data/tmp/timex/'
    
    fname = bufferDir + str(doc.get_doc_id()) + '.tml'
    if os.path.isfile(fname):
        f = open(fname, 'r')
        r = f.read()
        f.close()
    else:
        SERVER = 'http://localhost'
        PORT = 10001
        
        dct = utils.getDCT(doc)
        txt = utils.getRawTextSpecial(doc) # NB: this function adds two full stops to the text to separate first two sentences
        domain = "other" # newswire | narrative | other

        r = requests.post(url='%s:%d' % (SERVER, PORT), data={"query":txt, "dct":dct, "domain":domain})
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
        # get unicode string of annotated doc in TimeML format. It should only contain the TimeML(root),
        # DOCID, DCT and TEXT with TIMEX3 tags.
        r = r.json()['timeml']  # get the annotated text
        r = r.replace(' .', '', 2) # replace the first two occurences of the full stop that has been added earlier.
        
        f = open(fname, 'w')
        f.write(r.encode('ascii', 'ignore'))
        f.close()

    root = ET.fromstring(r.encode('ascii', 'ignore'))
    timex = root[2] # zoom on the TEXT tag
    
    # TBD: hacky
    # parse the contents of TEXT tag and record the location. Loop over the
    # TIMEX3 tags which should be the only tags here.
    w_count = 0
    if timex.text: # words before the first TIMEX3 tag
        #print timex.text
        prefix = timex.text.strip().split(' ') 
        w_count += len(prefix)
    for t in timex:
        tokens = t.text.strip().split(' ') # words inside the TIMEX3 tag
        #print tokens

        # get the tokens that correspond to this TIMEX3
        token_anchor_list = list()
        for i in range(0,len(tokens)):
            tmp = cat_parser.token_anchorType1()
            tmp.set_t_id(w_count + 1) # NB: token t_id are +1 wrt their index
            token_anchor_list.append(tmp)
            w_count += 1
        
        # build the TIMEX3 structure as in cat_parser
        wanted_type = ['DATE', 'TIME']
        if wanted_type.count(t.get('type')):
            timex3 = cat_parser.TIMEX3Type()
            timex3.set_functionInDocument('NONE')
            timex3.set_token_anchor(token_anchor_list) 
            timex3.set_value(t.get('value'))
            timex3.set_m_id(int(t.get('tid')[1:]))
            timex3.set_type(t.get('type'))
            # add the new TIMEX3 tag to the doc structure
            doc.Markables.TIMEX3.append(timex3)
    
        #utils.printTIMEX3(timex3)
        
        # Tail gives the words until the next tags
        if t.tail:
            tail = t.tail.strip().split(' ') # words after the TIMEX3 tag
            #print tail
            w_count = w_count+len(tail)
            #print w_count
        

    return doc
