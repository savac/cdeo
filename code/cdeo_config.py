import json

config_json_loc = '/home/sc/projects/cdeo/code/cdeo_config.json'

def setDefaults():
    defaults = {"restrict_entity_linking_to_sentence_flag": 1, 
    "stanfordcorenlp_loc": "/home/sc/work/stanford-corenlp-full-2015-04-20", 
    "n_epochs_entity": 15, 
    "n_epochs_timex": 15, 
    "levenshtein_threshold": 0.4, 
    "event_entity_link_threshold": 0.1, 
    "root_dir": "/home/sc/projects/cdeo/", 
    "uwtime_loc": "/home/sc/work/uwtime-standalone/target/uwtime-standalone-1.0.1.jar"}

    f = open(config_json_loc, 'w')
    json.dump(defaults, f)
    f.close()

def setConfig(field, val):
    f = open(config_json_loc, 'r')
    defaults = json.load(f)
    defaults[field] = val
    f.close()
    f = open(config_json_loc, 'w')
    json.dump(defaults, f)
    f.close()
        
def getConfig(field):
    f = open(config_json_loc, 'r')
    defaults = json.load(f)
    res = defaults[field]
    f.close()
    return res 
