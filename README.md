#### Cross-document Event Ordering

http://alt.qcri.org/semeval2015/task4/

##### Installation

1. Follow instructions from https://bitbucket.org/kentonl/uwtime-standalone to build UWTime
2. Download if required the Stanford CoreNLP (tested with version 3.5.2)
3. Update the fields "root_dir", "uwtime_loc", "stanfordcorenlp_loc" in code/cdeo_config.json with corresponding absolute paths.
4. Update variable <i>config_json_loc</i> in code/cdeo_config.py witht the absolute path to code/cdeo_config.json.
5. Install the python-levenshtein package. If you are using conda, run:

```conda install -c https://conda.anaconda.org/faircloth-lab python-levenshtein```

##### Running
It needs to run from the code/ directory

```cd code```

(Optional) If no preprocessed files exist in data/tmp/ we need start the UWTime server

```python -c "import cdeo; cdeo.startUWTimeServer()"```

To train on the Apple corpus (corpus 0) and test on the Airbus (corpus 1), GM (corpus 2) and Stock Markets (corpus 3) corpora run

```python -c "import cdeo; cdeo.run(test_corpus_list=[1,2,3], train_corpus_list=[0])"```

##### Results
|  | Airbus|GM|Stock|  |Total|  |
| --- | --- | --- | --- | --- | --- | --- |
|System|F1|F1|F1|P|R|F1|
|GPLSIUA_1|22.3|19.3|33.6|21.7|30.5|25.4|
|GPLSIUA_2|20.5|16.2|29.9|20.1|26.0|22.7|
|HeidelToul_1|19.6|7.3|20.4|20.1|14.8|17.0|
|HeidelToul_2|16.5|10.9|25.9|13.6|28.2|18.3|
|Thesis|25.5|25.8|36.6|29.8|28.2| 28.7|
|Paper|28.4|19.6|26.2|?|?|?|
