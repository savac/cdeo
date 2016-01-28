import cat_parser as cat_parser
import glob
import datetime
import numpy as np
import string
import re
import os
reload(cat_parser)

def loadDocs(inDir):
    files = glob.glob(inDir + '/*')
    
    collection = list()
    for f in files:
        with suppress_stdout_stderr():
            d = cat_parser.parse(f)
        if len(d.Markables.TIMEX3) == 1: # TBD:hacky
            d.Markables.set_DCT(d.Markables.TIMEX3[0])
        collection.append(d)    
    return collection

def loadEntityList(inFile):
    lst = list()
    f = open(inFile, 'r')
    while True:
        f_line = f.readline()
        if f_line == "":
            break
        lst.append(f_line.rstrip())
    return lst

def getDCT(doc):
    return doc.Markables.DCT.value
    
def getDoc(collection, docId):
    for doc in collection:
        if doc.get_doc_id() == docId:
            return doc
    
def getRawText(doc):
    raw = ''
    for t in doc.token:
        raw = raw + t.get_valueOf_() + ' '
    return raw
    
def getRawTextSpecial(doc):
    '''The title and the date will be separated from the rest of the sentences by the full stop.'''
    raw = ''
    old_sentence = 0
    for t in doc.token:
        new_sentence = t.sentence
        if (new_sentence == 1 and old_sentence == 0) or (new_sentence == 2 and old_sentence == 1):
            raw = raw + '.' + ' '
        old_sentence = new_sentence
        raw = raw + t.get_valueOf_() + ' '
    return raw
    
def getRawTextLines(doc):
    '''The title and the date will be separated from the rest of the sentences by the full stop.'''
    raw = ''
    old_sentence = 0
    for t in doc.token:
        new_sentence = t.sentence
        if new_sentence == old_sentence:
            raw = raw + t.get_valueOf_() + ' '
        else:
            raw = raw + '\n' + t.get_valueOf_() + ' '
            old_sentence = new_sentence
    return raw
        
def getRawTextList(doc):
    res = list()
    for t in doc.token:
        res.append(t.get_valueOf_())
    return res

def getRawTextList(doc):
    raw = []
    for t in doc.token:
        raw.append(t.get_valueOf_())
    return raw
    
def cleanFilesTML(inDir, outDir):
    collection = loadDocs(inDir)
    for doc in collection:
        f = open(outDir + '/' + str(doc.get_doc_id()) + '.txt', 'w')
        f.write(getRawText(doc).encode('ascii', 'ignore'))
        f.close()

def printTIMEX3(t):
    s = []
    for a in t.get_token_anchor():
        s.append(a.t_id)
    print t.value, t.functionInDocument, t.m_id, t.get_type(), 't_id = ', s

def getTIMEX3Text(doc, timex):
    '''Returns the text of the TIMEX3.'''
    s = []
    for a in timex.get_token_anchor():
        s.append(getToken(doc, a.t_id).valueOf_)
    return ' '.join(s)

def printEntity(t):
    s = []
    for a in t.get_token_anchor():
        s.append(a.t_id)
    print t.m_id, t.get_type(), 't_id = ', s
    
def getEvent(doc, m_id):
    for e in doc.Markables.EVENT_MENTION:
        if e.m_id == m_id:
            return e
            
def getEventText(doc, event):
    '''Returns the text of the event in a string converted to lower case.'''
    s = []
    for a in event.get_token_anchor():
        s.append(getToken(doc, a.t_id).valueOf_)
    if len(s) == 2 and s[0] == 'has':
        s = [s[1]]
    return '_'.join(s).lower()

def getEventTextFull(doc, event):
    '''Returns the text of the event in a string converted to lower case.'''
    s = []
    for a in event.get_token_anchor():
        s.append(getToken(doc, a.t_id).valueOf_)
    return '_'.join(s).lower()

def getTIMEX3(doc, m_id):
    for t in doc.Markables.TIMEX3:
        if t.m_id == m_id:
            return t
            
def getTIMEX3Stamp(doc, m_id): # TBD: this is a mess.
    if m_id == None:
        return 'XXXX-XX-XX'
    
    t = getTIMEX3(doc, m_id)
    
    # 2009-06-01T12:00
    if t.get_type() == 'TIME':
        tmp = t.value.split('T')
        return tmp[0]
    
    if t.value == 'PRESENT_REF':
        return doc.Markables.DCT.value
        
    elif t.value == 'PAST_REF' or t.value == 'FUTURE_REF':
        # <TIMEX3 tid="t7" type="DATE" value="PAST_REF">recently</TIMEX3> 
        return 'XXXX-XX-XX'
    
    else:
        val_split = t.value.split('-')
        
        # "2006-W13"
        if isnum(val_split[0]):
            if len(val_split) == 2:
                if val_split[1].count('W'):
                    week = int(val_split[1][1:])
                    date = datetime.date(int(val_split[0]), 1, 1) + datetime.timedelta((week-1)*7)
                    month = str(date.month)
                    if len(month) == 1:
                        month = '0' + month
                    return val_split[0] + '-' + month
        
        if not isnum(val_split[-1]): # return just the year
            if len(val_split[0]) == 4:
                return val_split[0] 
            else:
                return 'XXXX-XX-XX'
        else:
            if len(val_split[0]) == 4:
                return t.value
            else:
                return 'XXXX-XX-XX'
        
def compareTimestamps(ts0, ts1):
    # TBD: clean this up
    if ts0 == ts1:
        return True
        
    if ts0 == None or ts1 == None:
        return False
    
    res = True
    
    ts0 = ts0.split('-')
    ts1 = ts1.split('-')
    
    if ts0.count('XX'):
        ts0.remove('XX')
    if ts1.count('XX'):
        ts1.remove('XX')
    if ts0.count('XX'):
        ts0.remove('XX')
    if ts1.count('XX'):
        ts1.remove('XX')
    
    ts0.sort()
    ts1.sort()
        
    if not ts0 == ts1:
        res = False

    return res


def getEntity(doc, m_id):
    for e in doc.Markables.ENTITY_MENTION:
        if e.m_id == m_id:
            return e
            
def getEntityText(doc, entity):
    s = []
    for a in entity.get_token_anchor():
        s.append(getToken(doc, a.t_id).valueOf_)
    return ' '.join(s)
    
def getToken(doc, t_id):
    #print doc.get_doc_id(), t_id
    return doc.token[t_id - 1]
    
def getDoc(collection, doc_id):
    for doc in collection:
        if doc_id == doc.get_doc_id():
            return doc

def str2date(s):
    '''Transform a date in a format such as '2015-01-01' to Python's datetime.date type.
    Account for partial dates. Undefined parts of date will be assigned a value of 1.'''
    # TBD: these should not be grouped together
    #11	2009-06	127248-7-emphasized
    #11	2009-06-01	127227-7-give
    s = s.split('-')
    year = month = day = 1
    try:
        year = int(s[0])
        month = int(s[1])
        day = int(s[2])
    except (ValueError, IndexError):
        pass
    return datetime.date(year, month, day)
    
def isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def getNumericTokens(lst):
    '''Strip all non-numeric characters from the list of strings. Return list.'''
    res = list()
    for tok in lst:
        tok = re.sub('[^0-9]+', '', tok)
        if isnum(tok):
            res.append(tok)
    return res
            
#def myunique(lst):
#    '''NB:Order not preserved'''
#    d = {}
#    for l in lst:
#        d[l] = 1
#    return d.keys()
    
def myunique(seq):
   # not order preserving
   set = {}
   map(set.__setitem__, seq, [])
   return set.keys()    

def removePunctuation(s):
    table = string.maketrans("","")
    return s.translate(table, string.punctuation)
    
def removePunctuationList(lst):
    new_lst = []
    for t in lst:
        tclean = removePunctuation(t)
        if len(tclean)>0:
            new_lst.append(tclean)
    return new_lst
        
def idfScores(collection):
    '''Given a collection of documents, get a dictionary of idf scores using the classical formula.'''
    N = len(collection)
    res = dict()
    docList = list()
    for doc in collection:
        lst = getRawTextList(doc)
        lst = [tok.lower() for tok in lst]
        docList.append(lst)

    for lst in docList:
        for tok in lst:
            d = 0
            if not res.has_key(tok):
                for lst0 in docList:
                    if lst0.count(tok):
                        d += 1
                res[tok] = np.log(float(N)/d)
                
    return res
        
def goldTimelineList(targetEntityList, goldTimelineDir):
    '''Create a list of tuples (doc_it, sentence, event, entity). 
    Note entity is a list in case we have several events with the same name linked to different entitities.
    Eample:
    24	2011-01-17	17677-2-announced
    24	2011-01-17	17677-2-leave	17677-4-leave	18315-11-leave	17677-11-leave	18315-8-this	18315-8-leave'''
    # TBD: better to put this in dictionary of dictionaries for faster search later.
    res = list()
    for targetEntity in targetEntityList:
        fname = targetEntity.lower().replace(' ', '_')
        fname = fname.replace('&', '_and_')
        fname = goldTimelineDir + '/' + fname + '.txt'
        lines = [line.rstrip('\n') for line in open(fname, 'r')]
        for i in range(1, len(lines)): # skip 1st line
            tmp = lines[i].split('\t') # split on tabs
            tmpOrder = int(tmp[0])
            tmpDate = tmp[1]
            tmpList = tmp[2:] # get the '17677-2-leave	17677-4-leave	18315-11-leave' bit in a list
            for tmp in tmpList:
                tmp = tmp.split('-')
                docId = int(tmp[0])
                sen = int(tmp[1])
                #event = tmp[2]
                event = '-'.join(tmp[2:]) # deal with events like 'out-sold'
                res.append((docId, sen, event, targetEntity, tmpDate, tmpOrder))
    return res

def annotateCollectionWithGoldTimeline(collection, targetEntityList, goldTimelineDir):
    linksGold = goldTimelineList(targetEntityList, goldTimelineDir)
    #print linksGold
    
    # annotate event entries in docs
    for doc in collection:
        #print doc.get_doc_id()
        for ev in doc.Markables.EVENT_MENTION:
            event_t_id = ev.get_token_anchor()[0].t_id
            event_sentence = int(getToken(doc, event_t_id).sentence)
            event_text = getEventText(doc, ev).split('_')[0]
            #print (doc.get_doc_id(), event_sentence, event_text)
            #matches = [i for i, v in enumerate(linksGold) if v[0:3] == (doc.get_doc_id(), event_sentence, event_text)]
            matches = [i for i, v in enumerate(linksGold) if (v[0], v[1], v[2].lower().split('_')[0]) == (doc.get_doc_id(), event_sentence, event_text)] # match on the 1st event token
            if len(matches):
                goldEntity = linksGold[matches[0]][3] # just get the first one. Assume no events with the same name linked to different entities. 
                goldTimestamp = linksGold[matches[0]][4] 
                #print (doc.get_doc_id(),  event_sentence, event_text), goldEntity
                ev.set_linkedEntityName(goldEntity)
                
                # Clean this a little. The gold timeline has incomplete dates in different formats:
                # '2009' vs '2010-XX-XX'. Here we remove the Xs.
                
                ev.set_linkedTimestamp(goldTimestamp)


def mean_inc(val, mean_old, k):
    """Incremenal mean"""
    return mean_old + (val - mean_old)*1.0/k

def myargsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
