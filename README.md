#### Cross-document Event Ordering

http://alt.qcri.org/semeval2015/task4/

##### Installation

1. Follow instructions from https://bitbucket.org/kentonl/uwtime-standalone to build UWTime
2. Download if required the Stanford CoreNLP (tested with version 3.5.2)
3. Update the fields "root_dir", "uwtime_loc", "stanfordcorenlp_loc" in code/cdeo_config.json with corresponding absolute paths.
4. Update variable <i>config_json_loc</i> in code/cdeo_config.py witht the absolute path to code/cdeo_config.json.

##### Running
It needs to run from the code/ directory

```cd code```

(Optional) If no preprocessed files exist in data/tmp/ we need start the UWTime server

```python -c "import cdeo; cdeo.startUWTimeServer()"```

To train on the Apple corpus (corpus 0) and test on the Airbus corpus (corpus 1)

```python -c "import cdeo; cdeo.run(1, [0])"```

