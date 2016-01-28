import json

config_json_loc = '/home/sc15/projects/cdeo/code/cdeo_config.json'

def setDefaults():
    defaults = {
    'levenshtein_threshold' : 0.4,
     'event_entity_link_threshold' : 0.1
     }

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
