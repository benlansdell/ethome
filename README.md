[![codecov](https://codecov.io/gh/benlansdell/ethome/branch/master/graph/badge.svg?token=IJ0JJBOGGS)](https://codecov.io/gh/benlansdell/ethome)
![build](https://github.com/benlansdell/ethome/actions/workflows/workflow.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ethome-ml.svg)](https://badge.fury.io/py/ethome-ml)

# Ethome

Tools for machine learning of animal behavior. 

This library interprets pose-tracking files and behavior annotations to create features, train behavior classifiers, interpolate pose tracking data and other common analysis tasks. 

At present pose tracking data from DLC, SLEAP and NWB formats are supported, and behavior annotations from BORIS and NWB formats are supported.

## Features

* Read in animal pose data and corresponding behavior annotations to make supervised learning easy
* Scale data to desired physical units
* Interpolate pose data to improve low-confidence predictions 
* Create generic features for analysis and downstream ML tasks
* Create features specifically for mouse resident-intruder setup
* Quickly generate plots and movies with behavior predictions

## Installation

```
pip install ethome-ml
```

## Quickstart

It's easiest to start with an NWB file, which has metadata already connected to the pose data. 

Import
```python
from ethome import create_dataset
from ethome.io import get_sample_nwb_paths
```

Gather a sample NWB file
```python
fn_in = get_sample_nwb_paths()
```

Create the dataframe:
```python
dataset = create_dataset(fn_in)
```
`dataset` is an extended pandas DataFrame, so can be treated exactly as you would treat any other dataframe. `ethome` adds a bunch of metadata about the dataset, for instance you can list the body parts with:
```
dataset.pose.body_parts
```

A key functionality of `ethome` is the ability to easily create features for machine learning. You can use pre-built featuresets or make your own. For instance:
```python
dataset.features.add('distances')
``` 
will compute all distances between all body parts (both between and within animals).

There are featuresets specifically tailored for social mice studies (resident intruder). For instance, 
```
dataset.features.add('cnn1d_prob')
```
Uses a pretrained CNN to output probabilities of 3 behaviors (attack, mount, social investigation). For this, you must have labeled your body parts in a certain way (refer to How To). Other, more generic, feature creation functions are provided that work for any animal configuration. 

Now you can access a features table, labels, and groups for learning with `dataset.ml.features, dataset.ml.labels, dataset.ml.group`. From here it's easy to use some ML libraries to train a behavior classifier. For example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

cv = LeaveOneGroupOut()
model = RandomForestClassifier()
cross_val_score(model, 
                dataset.ml.features, 
                dataset.ml.labels, 
                groups = dataset.ml.group, 
                cv = cv)
```

Since the `dataset` object is just an extended Pandas dataframe we can manipulate it as such. E.g. we can add our model predictions to the dataframe:
```python
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(model, 
                                dataset.ml.features, 
                                dataset.ml.labels, 
                                groups = dataset.ml.group, 
                                cv = cv)
dataset['prediction'] = predictions
```

If the raw video file paths are provided in the metadata, under the `video` key, we can make a movie overlaying these predictions over the original video:
```python
dataset.io.save_movie(['label', 'prediction'], '.')
```
where `label` and `prediction` reference column names to annotate the video with.

A more detailed run through of features is provided in the How To guide.
 