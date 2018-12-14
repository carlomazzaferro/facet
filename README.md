facet
==============================

The scaffolding for this project was based on [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/)
with some twists. Namely, it adds tooling, configuration, and code to productionize the model as a web api,
as well as other utilites to generate reports, seriliaze models, and an expressive CLI.

Instructions
------------

First, install dependencies. This project uses python 3.6. Run:

```bash
$ pip install -r requirements.txt
```

And then, run 

```bash
$ pyhton setup.py install
```

Ideally this would be done from a clean virtualenv.

Analysis
--------

The analysis is self-contained in the Jupyter Notebook. Run 

```bash
$ jupyter lab
```

And navigate to the `notebooks` folder, and open `v1.ipynb`. It will contain all that is needed in what regards the
development of the model, as well as preprocessing of the data and hyperparameter optimization


The Library
-----------

One of my goals was creating a reproducible workflow, but also an engineering solution. As such, the project
comes with a fully featured CLI that will enable you to not only run all the analysis, but also server the models
developed as a web api. 

**Highlights**:

- Within the `src/` folder you'll find:
    - `/data` : utilities to generate the datasets for training and prediction
    - `/models` : our base model developed in the jupyter notebook, and utilities to train and predict using that model
    - `/prod` : a mini server based on the awesome library [`hug`](https://github.com/timothycrosley/hug)
    - `/transformers` : utilities to transform the dataset in an reusable manner -- based on the transformations developed in the notebook
    - `cli.py`: the definition of the `facet` cli

The CLI
-------

```bash
$ facet make-dataset --raw base.csv --train train.csv --test test.csv 
```

Will generate the train and test datasets, from the base dataset (provided by FW).


```bash
$ facet train --file train.csv     
```

Will train a model and apply all the transformations necessary to generate the very same output of the one from the
analysis. It will also store the model in `/models` and generate reports in `/reports`



```bash
$  facet predict --file test.csv
```

Will generate the output predictions and store them in `data/output`

The output file contains all the data that was not labeled, alongside the column 'Status', which contains probabilites
of class membership. The higher the value, the higher the probability of belonging to class 1. 


```bash
$  facet serve
```

Will start a server on port 8000 with an endpoint that takes in a JSON body and produces an output using the very same
model developed earlier. 
 
It can be tested easily, like this:

```bash
 curl -vX POST http://localhost:8000/predictions -d @sample1.json \                                                                                                                                                                                                                    ✘ 130 
     --header "Content-Type: application/json"
```
Which should return `{"prediction": 0.6}`. That is, a 60% probabilty of belonging to Status 1

And

```bash
 curl -vX POST http://localhost:8000/predictions -d @sample.json \                                                                                                                                                                                                                    ✘ 130 
     --header "Content-Type: application/json"
```
Which should return `{"prediction": 0.4}`. That is, a 40% probabilty of belonging to Status 1









