import utils.cat_parser as cat_parser
import utils.utils as utils
from datetime import date
import copy
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import itertools
import nltk.stem.porter as porter
import viterbi
import cdeo_config
reload(viterbi)
reload(utils)

def linkEventTIMEX3(doc, eventMIdList):
    # Do something simple. Link event to nearest TIMEX3 of the type DATE or PRESENT_REF.
    # Link only for those events with m_id from eventMIdList.
    res = list()
    wanted_type= ['DATE','TIME']
    for e in doc.Markables.EVENT_MENTION:
        if eventMIdList.count(e.m_id) > 0:
            event_t_id = e.get_token_anchor()[0].t_id
            d = len(doc.token)
            for t in doc.Markables.TIMEX3:
                if abs(event_t_id - t.get_token_anchor()[0].t_id) < d and (wanted_type.count(t.get_type())):
                    best_timex = t.m_id
                    d = abs(event_t_id - t.get_token_anchor()[0].t_id)
            
            # NB: deal with future dates wrt to dct: no event should be associated with a future date.
            this_date = utils.str2date(utils.getTIMEX3Stamp(doc, best_timex))
            dct_date = utils.str2date(doc.Markables.get_DCT().value)

            if this_date > dct_date:
                #print dct_date.timetuple(), this_date.timetuple()
                best_timex = doc.Markables.get_DCT().m_id # assing the m_id = 0, which should be the DCT

            res.append((e.m_id, best_timex)) 

    # return a list of of tuples containing m_id's: (event_m_id, timex3_m_id)
    return res

        
def linkEventTIMEX3ML(clf, doc, eventMIdList, syntactic_features):
    res = list()
    for event_m_id in eventMIdList:
        event = utils.getEvent(doc, event_m_id)
        predictedTIMEX = predictEventTIMEX3Link(clf, doc, event, syntactic_features)
        res.append((event_m_id, predictedTIMEX))
    return res

def trainEventTIMEX3Classifier(collection_train_list, syntactic_features):
    '''Takes a collection that has been annotated with the gold timeline. Returns a classifier for the event-timex3 links.'''
    
    featuresList = list()
    labelsList = list() 
    wanted_type = ['DATE', 'TIME']
    for tup in collection_train_list:
        (collection, targetEntityList) = tup
        for doc in collection:
            # get the list of all m_id's
            idsList = list()
            for t in doc.Markables.TIMEX3:
                if wanted_type.count(t.get_type()):
                    idsList.append(t.m_id)
            for event in doc.Markables.EVENT_MENTION:
                # identify the list of m_id's that correspond to the true timestamp
                goldTimestamp = event.get_linkedTimestamp()
                if goldTimestamp == 'XXXX-XX-XX': # we cannot find this timestamp in text
                    #print doc.get_doc_id(), utils.getEventText(doc, event), goldTimestamp
                    continue
                else:
                    event_t_id = event.get_token_anchor()[0].t_id
                    event_sentence = utils.getToken(doc, event_t_id).sentence
                    idGold = findGoldTIMEX3Id(doc, event, goldTimestamp)
                    if idGold == None:
                        #print '-----', doc.get_doc_id(), event_sentence, utils.getEventText(doc, event), goldTimestamp
                        continue

                    #print doc.get_doc_id(), utils.getEventText(doc, event), goldTimestamp

                    for m_id in idsList:
                        features = getLocalFeatures(doc, event, m_id, syntactic_features)
                        featuresList.append(features)
                        
                        if m_id == idGold:
                            labelsList.append(1)
                        else:
                            labelsList.append(0)
                    
    #clf = LogisticRegression(solver='lbfgs', C=0.9, penalty='l2', tol=1e-5, class_weight='auto', fit_intercept=True)
    #clf = LogisticRegression(solver='liblinear', C=1.0, penalty='l1', tol=1e-5, class_weight='auto', fit_intercept=True)
    n_iter = cdeo_config.getConfig('n_epochs_timex')
    clf = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=n_iter, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
    #clf = SVC(probability=True, kernel='rbf', class_weight='auto', tol=1e-4)
    clf.fit(featuresList, labelsList)

    print 'Event - timex features'
    print len(featuresList), np.sum(labelsList)

    return clf

def findGoldTIMEX3Id(doc, event, goldTimestamp):
    '''Return the m_id of the best matched TIMEX3 to the given timestamp'''
    event_t_id = event.get_token_anchor()[0].t_id
    event_sentence = utils.getToken(doc, event_t_id).sentence
    matchingTimexTokenList = list()
    best_m_id = None
    best_dist = len(doc.token)
    for t in doc.Markables.TIMEX3:
        timex_t_id =  t.get_token_anchor()[0].t_id
        timex_sentence = utils.getToken(doc, timex_t_id).sentence
        if utils.compareTimestamps(utils.getTIMEX3Stamp(doc, t.m_id), goldTimestamp):
            if event_sentence == timex_sentence: # if the gold timestamp exists in the event's that must be it
                return t.m_id 
            if abs(event_t_id - timex_t_id) < best_dist: # otherwise just get the nearest matching timestamp
                best_m_id = t.m_id
                best_dist = abs(event_t_id - timex_t_id)
    return best_m_id

def predictEventTIMEX3Link(clf, doc, event, syntactic_features):
    '''Given a trained classifier predict the TIMEX3 associated with the Event'''
    timexList = list()
    features = list()
    wanted_type = ['DATE', 'TIME']
    for timex in doc.Markables.TIMEX3:
        if wanted_type.count(timex.get_type()):
            features.append(getLocalFeatures(doc, event, timex.m_id, syntactic_features))
            timexList.append(timex.m_id)
            
    if len(features):
        scores = clf.decision_function(features)
        imax = np.argmax(scores) # get the highest timex with the highest confidence score
        return timexList[imax]
    else:
        return None
            
def extractSyntacticFeatures(collection):
    res = list()
    for doc in collection:
        for event in doc.Markables.EVENT_MENTION:
            event_t_id = event.get_token_anchor()[0].t_id
            event_sentence = utils.getToken(doc, event_t_id).sentence
            event_text = utils.getEventTextFull(doc, event)
            
            goldTimestamp = event.get_linkedTimestamp()
            if goldTimestamp == 'XXXX-XX-XX': # we cannot find this timestamp in text
                continue
            else:
                event_t_id = event.get_token_anchor()[0].t_id
                timex_m_id = findGoldTIMEX3Id(doc, event, goldTimestamp)
                if timex_m_id == None:
                    continue

            timex = utils.getTIMEX3(doc, timex_m_id)
            timex_t_id = timex.get_token_anchor()[-1].t_id # use the last token of the entity trigger
            timex_sentence = utils.getToken(doc, timex_t_id).sentence
            timex_text = utils.getTIMEX3Text(doc, timex)
            
            wanted_sentence = event_sentence 
            try:
                deps = doc.root[0][0][wanted_sentence][2] # NB: note the indexing! The title is a separate sentence in CAT but it's merged into 1st sentence in Stanford NLP parse.
            except IndexError:
                pass # TBD: should it not be 'continue' ?

            for dep in deps:
                if event_text.split('_').count(dep[0].text.lower()) and timex_text.split(' ').count(dep[1].text):
                    #print event_text + ',' + timex_text + ',' + dep.values()[0]
                    if res.count(dep.values()[0]) == 0:
                        res.append(dep.values()[0])
    #print res
    return res



def structuredPredictionTraining(collection_train_list, syntactic_features):
    '''Takes a collection that has been annotated with the gold timeline. Returns a classifier for the event-timex3 links.'''
    # initiate the weights vector
    w = np.zeros(510)
    wa = np.zeros(510)
    c = 1.0
    #lrate = 1
        
    trainError = list()
    wanted_type = ['DATE','TIME']
    print "Training Event to TIMEX linking model ..."
    for i in range(cdeo_config.getConfig('n_epochs_timex')): # number of iterations
        print 'Structured Perceptron Iteration: ', i
        #lrate = cdeo_config.getConfig('lr_decay')*lrate
        # do the prep
        # TBD: speed this up by saving
        for tup in collection_train_list:
            (collection, targetEntityList) = tup
            #random.shuffle(collection)
            for doc in collection:
                for targetEntity in targetEntityList:
                    #print doc.get_doc_id(), targetEntity
                    
                    # get the list of all TIMEX m_id's
                    allTimex = [None] # allow for a possibility of not matching with an actual timex 
                    for t in doc.Markables.TIMEX3:
                        if wanted_type.count(t.get_type()):
                            allTimex.append(t.m_id)
                    
                    # get lists of linked events and timex
                    linkedEvents = list()
                    linkedTimex = list()
                    for event in doc.Markables.EVENT_MENTION:
                        if event.get_linkedEntityName() == targetEntity:
                            goldTimestamp = event.get_linkedTimestamp()
                            if goldTimestamp == 'XXXX-XX-XX': # we also train on instances that have an undefined timestamp
                                #continue
                                linkedEvents.append(event)
                                linkedTimex.append(None)
                            else:
                                idGold = findGoldTIMEX3Id(doc, event, goldTimestamp)
                                if not idGold == None:  # only if we can find this timestamp in text
                                    linkedEvents.append(event)
                                    linkedTimex.append(idGold)


                    # for each document we have:
                    # - a list of events in linkedEvents
                    # - a corresponding list (training set) of linked timex m_id in linkedTimex
                    # - a list of all timex in allTimex
                    # - getLocalFeatures(doc, event, m_id, syntactic_features) will get the features for every event-timex pair
                    # - getGlobalFeatures(doc, (prev_event, t0), (event, t1)) will get the features for consecutive event-timex pairs
                    # - argmaxEventTIMEX(doc, event, allTimex, w)
                    
                    # precompute features
                    local_feat_dict = dict()
                    for event in linkedEvents:
                        for timex in allTimex:
                            local_feat_dict[(event, timex)] = getLocalFeatures(doc, event, timex, syntactic_features)
                            
                    global_feat_dict = dict()
                    for e in range(1, len(linkedEvents)):
                        prev_event = linkedEvents[e-1]
                        event = linkedEvents[e]
                        for t0 in allTimex:
                            for t1 in allTimex:
                                global_feat_dict[((prev_event, t0),(event, t1))] = getGlobalFeatures(doc, (prev_event, t0), (event, t1))
                    
                    if len(linkedEvents) and len(allTimex):
                        (linkedTimex_pred, pred) = argmaxEventTIMEX(doc, linkedEvents, allTimex, w, local_feat_dict, global_feat_dict)
                                                
                        if not tuple(linkedTimex) == linkedTimex_pred:
                            w = w + (getPHI(doc, linkedEvents, linkedTimex, local_feat_dict, global_feat_dict) - getPHI(doc, linkedEvents, linkedTimex_pred, local_feat_dict, global_feat_dict))
                        wa = wa + w
                        c += 1
    return wa/c

def getPHI(doc, listEvents, listTimex, local_feat_dict, global_feat_dict):
    PHI = np.zeros(500)
    
    # local features
    for event, timex in zip(listEvents, listTimex):
        PHI += local_feat_dict[(event, timex)]
        
    # global features
    PHI2 = np.zeros(10)
    for ind in range(1, len(listEvents)):
        prev_event = listEvents[ind-1]
        prev_timex = listTimex[ind-1]
        event = listEvents[ind]
        timex = listTimex[ind]
        PHI2 += global_feat_dict[((prev_event, prev_timex),(event, timex))]
        
    return np.append(PHI, PHI2)

def structuredPrediction(w, doc, listEvents, syntactic_features):
    '''Given the weights from the structured perceptron predict the TIMEX3 associated with the Event'''
    # get timex list
    timexList = [None] # allow for a possibility of no match
    wanted_type = ['DATE', 'TIME']
    for timex in doc.Markables.TIMEX3:
        if wanted_type.count(timex.get_type()):
            #if not utils.getTIMEX3Stamp(doc, timex.m_id) == 'XXXX-XX-XX': # TBD: is this right?
            timexList.append(timex.m_id)
            
    # precompute features
    local_feat_dict = dict()
    for event in listEvents:
        for timex in timexList:
            local_feat_dict[(event, timex)] = getLocalFeatures(doc, event, timex, syntactic_features)
    
    global_feat_dict = dict()
    for e in range(1, len(listEvents)):
        prev_event = listEvents[e-1]
        event = listEvents[e]
        for t0 in timexList:
            for t1 in timexList:
                global_feat_dict[((prev_event, t0),(event, t1))] = getGlobalFeatures(doc, (prev_event, t0), (event, t1))
    
    (best_seq, best_pred) = argmaxEventTIMEX(doc, listEvents, timexList, w, local_feat_dict, global_feat_dict)

    return best_seq

def linkEventTIMEX3SP(w, doc, eventMIdList, syntactic_features):
    res = list()
    listEvents = list()
    
    if len(eventMIdList):
        for event_m_id in eventMIdList:
            listEvents.append(utils.getEvent(doc, event_m_id))
        
        predictedTIMEX_list = structuredPrediction(w, doc, listEvents, syntactic_features)
            
        for i in range(0, len(eventMIdList)):
            res.append((eventMIdList[i], predictedTIMEX_list[i]))

    return res
    
def getGEN(len_events, allTimex):
    return list(itertools.product(allTimex, repeat=len_events))

def argmaxEventTIMEX(doc, linkedEvents, allTimex, w, local_feat_dict, global_feat_dict):
    '''Find the argmax'''
    
    allTimexLists = [copy.deepcopy(allTimex) for i in range(len(linkedEvents))]
    
    ew = w[0:500]
    tw = w[500:510]
    hmm = viterbi.hmmClass(global_feat_dict, local_feat_dict, tw, ew)
    
    thisViterbi = viterbi.Viterbi(hmm, linkedEvents, allTimexLists)
    best_seq = thisViterbi.return_max()
    
    return (best_seq, 0)

def getLocalFeatures(doc, event, timex_m_id, syntactic_features):
    '''Get the feature vector for the event and the target entity.
    Feature 1: abs(distance in tokens)
    Feature 2: abs(distance in sentences)
    ....
    '''
    features = np.array([0]*500) # features
    
    if timex_m_id == None:
        features[0] = 1
        return features
    ind = 1
    
    event_t_id = event.get_token_anchor()[0].t_id
    event_sentence = utils.getToken(doc, event_t_id).sentence    
    event_text = utils.getEventTextFull(doc, event)
    timex = utils.getTIMEX3(doc, timex_m_id)
    timex_t_id = timex.get_token_anchor()[0].t_id # use the first token of the timex trigger
    timex_sentence = utils.getToken(doc, timex_t_id).sentence
    timex_text = utils.getTIMEX3Text(doc, timex)
    timex_timestamp = utils.getTIMEX3Stamp(doc, timex_m_id)
            
    # relative distance in sentences:
    d = event_sentence - timex_sentence
    n = 10
    if abs(d) <= n:
        features[ind+d+n] = 1
    ind += 2*n+1
    
    # bins of 5 unit token distance
    d = event_t_id - timex_t_id
    n = 40
    bin_size = 5.0
    if abs(d) <= n:
        thisBin = int(np.fix(d/bin_size))
        features[ind+thisBin+int(n/bin_size)] = 1
    ind += 2*(n/bin_size)+1    

    # bins of 10 unit token distance
    d = event_t_id - timex_t_id
    n = 50
    bin_size = 10.0
    if abs(d) <= n:
        thisBin = int(np.fix(d/bin_size))
        features[ind+thisBin+int(n/bin_size)] = 1
    ind += 2*(n/bin_size)+1

    # bins of 20 unit token distance
    d = event_t_id - timex_t_id
    n = 60
    bin_size = 20.0
    if abs(d) <= n:
        thisBin = int(np.fix(d/bin_size))
        features[ind+thisBin+int(n/bin_size)] = 1
    ind += 2*(n/bin_size)+1

    
    # if mentioned before the event
    if timex_t_id < event_t_id:
        features[ind] = 1
    ind += 1

    # syntactic features
    if event_sentence == timex_sentence:
        wanted_sentence = event_sentence
        deps = doc.root[0][0][wanted_sentence][2] # NB: note the indexing! The title is a separate sentence in CAT but it's merged into 1st sentence in Stanford NLP parse.

        # TBD: networkx to find the paths in a network
        for dep in deps:
            if event_text.split('_').count(dep[0].text.lower()) and timex_text.split(' ').count(dep[1].text):
                for i in range(0, len(syntactic_features)):
                    if dep.values()[0] == syntactic_features[i]:
                        #print  dep[1].text, dep[0].text, ':' + syntactic_features[i], event_sentence, doc.get_doc_id()
                        features[ind+i] = 1
    ind += len(syntactic_features)

    # check if date is in the future
    this_date = utils.str2date(utils.getTIMEX3Stamp(doc, timex_m_id))
    dct_date = utils.str2date(doc.Markables.get_DCT().value)
    if this_date > dct_date:
        features[ind] = 1
    ind += 1

    # if the timestamp is undefined
    if timex_timestamp == 'XXXX-XX-XX':
        features[ind] = 1
    ind += 1

    # if timestamp is complete or incomplete
    tmp = timex_timestamp.split('-')
    if len(tmp) == 3:
        if utils.isnum(tmp[0]) and utils.isnum(tmp[1]) and utils.isnum(tmp[2]):
            features[ind] = 1
    ind += 1
    if len(tmp) == 2:
        if utils.isnum(tmp[0]) and utils.isnum(tmp[1]):
            features[ind] = 1
    ind += 1
    if len(tmp) == 1:
        if utils.isnum(tmp[0]):
            features[ind] = 1
    ind += 1

    return features

def getGlobalFeatures(doc, t0, t1):
    '''t0 and t1 are adjacent hidden variables in a HMM representing a (event, timex) tuple.'''
    features = np.zeros(10)
    stemmer = porter.PorterStemmer()
    
    (event0, timex0) = t0
    (event1, timex1) = t1
    
    event_t_id = event0.get_token_anchor()[0].t_id
    event_sentence0 = utils.getToken(doc, event_t_id).sentence
    
    event_t_id = event1.get_token_anchor()[0].t_id
    event_sentence1 = utils.getToken(doc, event_t_id).sentence
    
    str_event0 = utils.getEventTextFull(doc, event0)
    str_event1 = utils.getEventTextFull(doc, event1)
    
    stem0 = stemmer.stem(str_event0.split('_')[0])
    stem1 = stemmer.stem(str_event1.split('_')[0])

    
    # just timex
    if timex0 == timex1:
        features[0] = 1
    ind = 1

    if not timex0 == timex1:
        features[ind] = 1
    ind += 1
    
    # timex and event string
    if timex0 == timex1 and str_event0 == str_event1:
        features[ind] = 1
    ind += 1

    # timex and event stem
    if timex0 == timex1 and stem0 == stem1:
        features[ind] = 1
    ind += 1

    # timex and sentence
    if timex0 == timex1 and event_sentence0 == event_sentence1:
        features[ind] = 1
    ind += 1

    # talking words
    talking_words = ['said','presented','explained','proposed','stated','thanked','refused','described','pointed','responded','anticipated','concluded', 'called', 'praised']
    # timex and event string type
    if timex0 == timex1 and talking_words.count(str_event0) and talking_words.count(str_event1):
        features[ind] = 1
    ind += 1

    return features
    
