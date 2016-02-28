#### Cross-document Event Ordering

http://alt.qcri.org/semeval2015/task4/

##### Installation

1. Follow instructions from https://bitbucket.org/kentonl/uwtime-standalone to build UWTime*
2. Download if required the Stanford CoreNLP (tested with version 3.5.2)
3. Update the fields "root_dir", "uwtime_loc", "stanfordcorenlp_loc" in code/cdeo_config.json with corresponding absolute paths.
4. Update variable <i>config_json_loc</i> in code/cdeo_config.py witht the absolute path to code/cdeo_config.json.
5. Install the python-levenshtein package. If you are using conda, run:
```conda install -c https://conda.anaconda.org/faircloth-lab python-levenshtein```

*The directories data/tmp/ner and data/tmp/timex already contain the results of the UWTime and Stanford Core NLP processings so (1) and (2) are only required if planning to re-run the timex identification and parsing/coref.

##### Running
It needs to run from the code/ directory

```cd code```

(Optional) If no preprocessed files exist in data/tmp/ we need start the UWTime server

```python -c "import cdeo; cdeo.startUWTimeServer()"```

To train on the Apple corpus (corpus 0) and test on the Airbus (corpus 1), GM (corpus 2) and Stock Markets (corpus 3) using the structured perceptron algorithm run:

```python -c "import cdeo; cdeo.run(test_corpus_list=[1,2,3], train_corpus_list=[0], link_model='structured_perceptron')"```

To use the perceptron algorithm run:

```python -c "import cdeo; cdeo.run(test_corpus_list=[1,2,3], train_corpus_list=[0], link_model='perceptron')"```

To run cross-validation (hold out one target entity, train on the rest, predict the timeline for the held out entity):

```python -c "import cdeo; cdeo.crossval(train_corpus=0, link_model='structured_perceptron')"```

Hyperparamenter tuning for the number of Event-Timex and Event-Entity iterations. You'll need to edit the code to change ranges (both currently [5,10,15,20,25]):

```python -c "import cdeo; cdeo.tuning('structured_percetron')"```

To get the total micro scores place all predicted and gold timelines in two separate folders. The run the following after adjusting the paths:

```python evaluation_all.py ~/projects/cdeo/data/evaluation/combined/gold/ ~/projects/cdeo/data/evaluation/combined/results_structured/```

##### Results
|  | Airbus|GM|Stock|  |Total|  |
| --- | --- | --- | --- | --- | --- | --- |
|System|F1|F1|F1|P|R|F1|
|GPLSIUA_1|22.3|19.3|33.6|21.7|30.5|25.4|
|GPLSIUA_2|20.5|16.2|29.9|20.1|26.0|22.7|
|HeidelToul_1|19.6|7.3|20.4|20.1|14.8|17.0|
|HeidelToul_2|16.5|10.9|25.9|13.6|28.2|18.3|
|Thesis|25.5|25.8|36.6|29.8|28.2| 28.7|
|Our_System_Binary|17.99|20.97|34.95|25.97|24.79|25.37|
|Our_System_Alignment|25.65|26.64|32.35|29.05|28.12|28.58|
