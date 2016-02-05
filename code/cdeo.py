import utils.cat_parser as cat_parser
import utils.utils as utils
import subprocess
import numpy as np
from datetime import date
import sys
import os
import re
import EntityExtraction
import TimestampExtraction
import EventEntityLinking
import EventTIMEX3Linking
import Ordering
import nltk.stem.porter as porter
import cdeo_config
reload (cdeo_config)
reload(cat_parser)
reload(utils)
reload(EntityExtraction)
reload(TimestampExtraction)
reload(EventEntityLinking)
reload(EventTIMEX3Linking)
reload(Ordering)

# init stemmer
stemmer = porter.PorterStemmer()

def startUWTimeServer():
    h = subprocess.Popen(['java','-jar',cdeo_config.getConfig("uwtime_loc")])
    return h

def getCollection(ncorpus):
    print '\n=============================================================='
    print 'Reading in corpus %d...' %(ncorpus)
    print '=============================================================='

    corpus = ncorpus
    inFileEntity = '../data/evaluation/corpus_%s/list_target_entities_corpus_%s.txt' %(corpus, corpus)
    inDirCAT= '../data/evaluation/corpus_%s/corpus_trackB_CAT' %(corpus)
    
    targetEntityList = utils.loadEntityList(inFileEntity)
    collection = utils.loadDocs(inDirCAT)
    
    # annotate timestamps and target entities
    collection0 = []
    
    for indoc in collection:
        doc = TimestampExtraction.getTimestamps(indoc) # NB: doc 120578 has a mistake in the DCT annotation. The CAT files has been corrected by hand.
        doc = EntityExtraction.getEntitiesStanfordNLP(doc, targetEntityList)
        # NB: sort the events in the order that they appear in the text i.e. sort by token position
        oldList = list()
        for event in doc.Markables.EVENT_MENTION:
            event_t_id = event.get_token_anchor()[0].t_id
            oldList.append(event_t_id)
        indSorted = utils.myargsort(oldList)
        newList = [doc.Markables.EVENT_MENTION[i] for i in indSorted]
        doc.Markables.EVENT_MENTION = newList
        
        collection0.append(doc)
        
    collection = collection0
    return (collection, targetEntityList)

def evaluateEventEntityLinking(timeline, corpus, collection):
    inFileEntity = '../data/evaluation/corpus_%s/list_target_entities_corpus_%s.txt' %(corpus, corpus)
    goldTimelineDir = '../data/evaluation/corpus_%s/TimeLines_Gold_Standard' %(corpus) # TBD: clean this up
    targetEntityList = utils.loadEntityList(inFileEntity)
    goldList = utils.goldTimelineList(targetEntityList, goldTimelineDir)

    sys.stdout.write('\nEvent to Entity Linking Evaluation (Entity, Recall, Precision, F1-score)\n')

    totalList = list()
    timeList = list()
    for targetEntity in timeline.keys():
        # [doc.get_doc_id(), event_m_id, timex3_m_id, this_date, 1, event_sentence, str_event, event_stem]
        
        thisEntity = list()
        for lst in timeline[targetEntity]:
            [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = lst
            thisEntity.append((docId, event_sentence, str_event, targetEntity))
            
            doc = utils.getDoc(collection, docId)
            date_str = utils.getTIMEX3Stamp(doc, timex3_m_id)
            timeList.append((docId, event_sentence, str_event, targetEntity, date_str))

        thisEntityGold = [x[0:4] for x in goldList if x[3] == targetEntity] # filter for this entity and discard timex and order
        try:
            precision = float(len(set(thisEntity).intersection(thisEntityGold)))/len(thisEntity) * 100
            recall = float(len(set(thisEntity).intersection(thisEntityGold)))/len(thisEntityGold) * 100
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1 = 0
        sys.stdout.write(('%-30s\t%3.9f\t%3.9f\t%3.9f\n' %(targetEntity, recall, precision, f1) )) 
        
        totalList += thisEntity
    try:
        tmpGoldList = [x[0:4] for x in goldList] # discard timex and order
        precision = float(len(set(totalList).intersection(tmpGoldList)))/len(totalList) * 100
        recall = float(len(set(totalList).intersection(tmpGoldList)))/len(goldList) * 100
        f1 = 2 * precision * recall / (precision + recall)
        
        # the number matching event-times out of the number of matching event-entity        
        tmp0 = [(x[0], x[1], x[2], x[3], x[4]) for x in goldList] # keep timex and event and target entity
        tt = float(len(set(timeList).intersection(tmp0)))
        tmp1 = [x[0:4] for x in goldList] # discard timex and order
        tp = float(len(set(totalList).intersection(tmp1)))
        acc = tt/tp*100
        
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f1 = 0
        acc = 0
    sys.stdout.write(('%-30s\t%3.9f\t%3.9f\t%3.9f\n' %('Total', recall, precision, f1) )) 
    sys.stdout.write('Total events: %f\n' %(len(goldList)))
    sys.stdout.write('TP (number of correct event - entity): %f\n' %(tp))
    sys.stdout.write('TT (number of correct event - time): %f\n' %(tt))

    sys.stdout.write('Accuracy (matching event-times over matching event-entity numbers): %f\n' %(acc))


def evaluateEventTIMEX3Linking(timeline, corpus, collection):
    inFileEntity = '../data/evaluation/corpus_%s/list_target_entities_corpus_%s.txt' %(corpus, corpus)
    goldTimelineDir = '../data/evaluation/corpus_%s/TimeLines_Gold_Standard' %(corpus) # TBD: clean this up
    targetEntityList = utils.loadEntityList(inFileEntity)
    goldList = utils.goldTimelineList(targetEntityList, goldTimelineDir) # ((docId, sen, event, targetEntity, tmpDate, tmpOrder))

    sys.stdout.write('\nEvent to Timestamp Evaluation (Entity, Recall, Precision, F1-score)\n')

    totalList = list()
    for targetEntity in timeline.keys():        
        thisEntity = list()
        for lst in timeline[targetEntity]:
            [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = lst
            doc = utils.getDoc(collection, docId)
            date_str = utils.getTIMEX3Stamp(doc, timex3_m_id)
            totalList.append((docId, event_sentence, str_event, date_str))
    try:
        tmpGoldList = [(x[0], x[1], x[2], x[4]) for x in goldList] # keep timex and event
        precision = float(len(set(totalList).intersection(tmpGoldList)))/len(totalList) * 100
        recall = float(len(set(totalList).intersection(tmpGoldList)))/len(goldList) * 100
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f1 = 0
        
    print len(goldList), len(totalList)
    sys.stdout.write(('%-30s\t%3.9f\t%3.9f\t%3.9f\n' %('Total', recall, precision, f1) )) 


def run(test_corpus_list = [1], train_corpus_list = [0], ee_link_model='seq', et_link_model='seq'):
    ''' 
    Inputs 
    test_corpus_list - list of index numbers for the test corpora ( Apple corpus (corpus 0) and test on the Airbus (corpus 1), GM (corpus 2) and Stock Markets (corpus 3) )
    train_corpus_list - as above for training
    ee_link_model - Event to Entity linking model type. valid options are 'maxent' (MaxEnt model) and 'seq' (Sequence model)
    et_link_model - Event to Timestamp
    This scripts runs all the steps of timeline extraction:
    1. Extract timestamps
    2. Extract target entities
    3. Link events to target entities
    4. Link events and timestamps
    5. Order
    '''

    # Read in the training corpus 
    collection_train_list = list()
    entity_syntactic_features = list()
    timex_syntactic_features = list()
    for train_corpus in train_corpus_list:
        (collection_train, targetEntityList_train) = getCollection(train_corpus)
        goldTimelineDir = '../data/evaluation/corpus_%s/TimeLines_Gold_Standard' %(train_corpus) # TBD: clean this up
        utils.annotateCollectionWithGoldTimeline(collection_train, targetEntityList_train, goldTimelineDir)
        collection_train_list.append((collection_train, targetEntityList_train))
        
        entity_syntactic_features += EventEntityLinking.extractSyntacticFeatures(collection_train)
        timex_syntactic_features += EventTIMEX3Linking.extractSyntacticFeatures(collection_train)

    # Train the link classifiers
    if ee_link_model == 'maxent':
        clfEntity = EventEntityLinking.trainEventEntityClassifier(collection_train_list, entity_syntactic_features)  
    elif ee_link_model == 'seq':
        w_ee = EventEntityLinking.structuredPredictionTraining(collection_train_list, entity_syntactic_features)   
        
    if et_link_model == 'maxent':
        clfTIMEX3 = EventTIMEX3Linking.trainEventTIMEX3Classifier(collection_train_list, timex_syntactic_features)
    elif et_link_model == 'seq':
        w = EventTIMEX3Linking.structuredPredictionTraining(collection_train_list, timex_syntactic_features)

    #print w_ee
    #print w

    # Read in, predict and evaluate the test corpus
    # TBD: refactor
    for corpus in test_corpus_list:
        (collection, targetEntityList) = getCollection(corpus)
        resDir = '../data/evaluation/corpus_%s/results/' %(corpus)
        timeline = dict()
        print 'Predicting timeline ...'
        # We create a dictionary to hold all the extracted linked events, entities and timestamps:
        # Keys are target entities.
        # Values are lists of tuples (doc_id, event_m_id, timex3_m_id, date)
        for doc in collection:
            if ee_link_model == 'maxent':
                dictEventEntity = EventEntityLinking.linkEventEntityML(clfEntity, doc, targetEntityList, entity_syntactic_features)
            elif ee_link_model == 'seq':
                dictEventEntity = EventEntityLinking.linkEventEntitySP(w_ee, doc, targetEntityList, entity_syntactic_features)
            for targetEntity in targetEntityList:
                # legacy role based
                #listEventEntity = EventEntityLinking.linkEventEntityRuleBased(doc, targetEntity) # returns list of event m_id's
                #listEventTIMEX3 = EventTIMEX3Linking.linkEventTIMEX3(doc, listEventEntity)

                listEventEntity = dictEventEntity[targetEntity]
                
                if et_link_model == 'maxent':
                    listEventTIMEX3 = EventTIMEX3Linking.linkEventTIMEX3ML(clfTIMEX3, doc, listEventEntity, timex_syntactic_features)
                elif et_link_model == 'seq':
                    listEventTIMEX3 = EventTIMEX3Linking.linkEventTIMEX3SP(w, doc, listEventEntity, timex_syntactic_features)
                
                if not timeline.has_key(targetEntity):
                    timeline[targetEntity] = list()
                for event_time in listEventTIMEX3:
                    (event_m_id, timex3_m_id) = event_time
                    this_date = utils.str2date(utils.getTIMEX3Stamp(doc, timex3_m_id))

                    event = utils.getEvent(doc, event_m_id)
                    event_t_id = event.get_token_anchor()[0].t_id
                    event_sentence = utils.getToken(doc, event_t_id).sentence
                    
                    str_event = []
                    for a in event.get_token_anchor():
                        str_event.append(doc.token[a.t_id-1].get_valueOf_()) # TBD: better to look for tokens searching by t_id.
                    event_stem = stemmer.stem(str_event[0])
                    str_event = '_'.join(str_event)
                    str_event = str_event.lower()
                    
                    order = 1
                    eventClusterId = 0
                    lst = [doc.get_doc_id(), event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] # assign an order number of 1 for now
                    timeline[targetEntity].append(lst)
        # Given the dictionary from above we order the tuples as sepcified by the CDEO task
        timeline = Ordering.order(collection, timeline)
        
        # printing and saving to file the predicted timelines
        for targetEntity in timeline.keys():
            fname = targetEntity.lower()
            fname = fname.replace('&', '_and_')
            fname = resDir + re.sub('[^0-9a-z]+', '_', fname) + '.txt'
            f = open(fname, 'w')
            f.write(targetEntity)    
            
            last_eventClusterId = -1
            for lst in timeline[targetEntity]:
                [doc_id, event_m_id, timex3_m_id, this_date, order, nsentence, str_event, this_stem, eventClusterId] = lst

                doc = utils.getDoc(collection, doc_id)
                
                date_str = utils.getTIMEX3Stamp(doc, timex3_m_id)

                # docid-sentence, event, timex3, timex3 text
                if eventClusterId == last_eventClusterId: # TBD: this is flaky as we only check events adjacent in timeline
                    f.write('\t' + str(doc.doc_id) + '-' + str(nsentence)+ '-' + str_event) #+  "\t('" + str_timex + "')" 
                else:
                    f.write('\n' + str(order) + '\t' + date_str + '\t' + str(doc_id) + '-' + str(nsentence)+ '-' + str_event) #+  "\t('" + str_timex + "')" 
                last_eventClusterId = eventClusterId
                last_stem = this_stem
            f.close()

        # Evaluation
        evaluateEventEntityLinking(timeline, corpus, collection)
        evaluateEventTIMEX3Linking(timeline, corpus, collection)
        try:
            print "\nEnd to end evaluation"
            os.chdir('evaluation_tool')
            os.system(('python evaluation_all.py ../../data/evaluation/corpus_%d/TimeLines_Gold_Standard/ ../../data/evaluation/corpus_%d/results/' %(corpus,corpus)))
            #os.system(('python evaluation_all.py --ord ../../evaluation/corpus_%d/TimeLines_Gold_Standard/ ../../evaluation/corpus_%d/results/' %(corpus,corpus)))
            os.chdir('..')
        except:
            print 'Evaluation error'
