import os
# Generate data types and parser for CAT and (modified) TimeML formats
os.system('../../tools/generateDS-2.16a0/generateDS.py -o cat_parser.py ../../data/CAT/cat_schema.xsd')
os.system('../../tools/generateDS-2.16a0/generateDS.py -o timeml_parser.py ../../data/TimeML/timeml_schema.xsd')
