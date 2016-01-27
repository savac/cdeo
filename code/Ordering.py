import utils.cat_parser as cat_parser
import utils.utils as utils
from datetime import date
import copy
import random
import numpy as np
reload(utils)
 
def order(collection, timeline):
    '''Take a list of [doc.get_doc_id(), event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem] '''      
    # cluster id
    # eventClusterId
    # look at dobj and ccomp dependencies
    # we need to establish given a two or more events with the identical timestamp if:
    # 1. The events are simultaneous: assign the same order number
    # 2. The events are identical: assign the same eventClusterId
    
    # create a dictionary with the timestamp as key
    ordered_timeline = dict()
    for targetEntity in timeline.keys():
        ordered_timeline[targetEntity] = list()
        dateDict = dict()
        
        # split events to groups that contain the same timestamp
        for lst in timeline[targetEntity]:
            [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = lst
            if dateDict.has_key(this_date):
                dateDict[this_date].append(lst)
            else:
                dateDict[this_date] = [lst]
        
        # easy to sort out the dates straight away
        keysSorted = dateDict.keys()
        keysSorted.sort()

        # order and cluster here
        offsetOrder = 0 if keysSorted.count(date(1,1,1)) else 1 # unknown dates 'xxxx-xx-xx' start at 0
        indOrder = 0
        indClusterId = 0
        for thisDate in keysSorted: # loop through sorted dates
            thisGroup = dateDict[thisDate]
            
            # by sentence
            thisGroup.sort(key=lambda tup: tup[5])
            # by stem
            thisGroup.sort(key=lambda tup: tup[7])      
            
            thisGroup2 = list()
            newOrder = newEventClusterId = 0 # variable names are a bit of a mess here
            
            # very first entry
            [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = thisGroup[0]
            thisGroup2.append([docId, event_m_id, timex3_m_id, this_date, newOrder, event_sentence, str_event, event_stem, newEventClusterId])
            
            distinct_events = ['said', 'announc', 'state', 'describ', 'seek', 'call']
            for i in range(1, len(thisGroup)):
                [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = thisGroup[i]

                # deal with unknown dates first
                if this_date == date(1,1,1):
                    if (not thisGroup[i-1][7] == thisGroup[i][7]) or distinct_events.count(thisGroup[i][7]): # if stems are different increment cluster id
                        newEventClusterId += 1
                    thisGroup2.append([docId, event_m_id, timex3_m_id, this_date, newOrder, event_sentence, str_event, event_stem, newEventClusterId])
                    continue
                    
                # Compare stems. Cluster together those events with the same stems except for those specified in distinct_events and from the same document
                if (not thisGroup[i-1][7] == thisGroup[i][7]) or distinct_events.count(thisGroup[i][7]):
                    newEventClusterId += 1
                    
                thisGroup2.append([docId, event_m_id, timex3_m_id, this_date, newOrder, event_sentence, str_event, event_stem, newEventClusterId])
                
            
            # This just gets the right absolute order number
            for lst in thisGroup2:
                [docId, event_m_id, timex3_m_id, this_date, order, event_sentence, str_event, event_stem, eventClusterId] = lst
                order += indOrder
                eventClusterId += indClusterId
                new_lst = [docId, event_m_id, timex3_m_id, this_date, order + offsetOrder, event_sentence, str_event, event_stem, eventClusterId]
                ordered_timeline[targetEntity].append(new_lst)
            indOrder = order + 1
            indClusterId = eventClusterId + 1
    return ordered_timeline
    
def myargsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
