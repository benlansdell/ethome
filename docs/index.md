[![codecov](https://codecov.io/gh/benlansdell/ethome/branch/master/graph/badge.svg?token=IJ0JJBOGGS)](https://codecov.io/gh/benlansdell/ethome)
![build](https://github.com/benlansdell/ethome/actions/workflows/workflow.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ethome-ml.svg)](https://badge.fury.io/py/ethome-ml)

# Ethome

Tools for machine learning of animal behavior.

This library interprets pose-tracking files (at present, from DLC or NWB formats) and behavior annotations (at present, from BORIS and NWB formats) to help train a behavior classifier, perform unsupervised learning, and other common analysis tasks. 

## Features

* Read in pose data and corresponding behavior annotations to make supervised learning easy
* Interpolate pose data to improve low-confidence predictions 
* Create generic features for analysis and downstream ML tasks
* Create features specifically for mouse resident-intruder setup
* Perform unsupervised learning on pose data to extract discrete behavioral motifs (similar to MotionMapper)
* Quickly generate a movie with behavior predictions

## Installation

```
pip install ethome-ml
```

## Quickstart

It's easiest to start with an NWB file, which has metadata already connected to the pose data. 

Import
```python
from ethome import create_dataset, create_metadata
from ethome.io import get_sample_data_paths_dlcboris
```

Gather a sample NWB file
```python
tracking_files, annotation_files = get_sample_data_paths_dlcboris()
```

Create the dataframe:
```python
metadata = create_metadata(tracking_files, labels = annotation_files, fps = 30)
animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
dataset = create_dataset(metadata, animal_renamer = animal_renamer)
```
`dataset` is an extended pandas DataFrame, so can be treated exactly as you would treat any other dataframe. `ethome` adds a bunch of metadata about the dataset, for instance you can list the body parts with:
```
dataset.pose.body_parts
```

The key functionality of `ethome` is the ability to easily create features for machine learning. You can use pre-built featuresets, or make your own. Here are two designed specifically to work with a mouse resident-intruder setup:
```python
dataset.features.add('mars')
```
You must have labeled your body parts in a certain way to use these two feature sets (see the How To). But other, more generic, feature creation functions are provided that work for any animal configuration. (The 'mars' feature-set is designed for studying social behavior in mice, based heavily on the MARS framework Segalin et al. [1])

Now you can access a features table, labels, and groups for learning with `dataset.ml.features, dataset.ml.labels, dataset.ml.group`. From here it's easy to use some ML libraries to train a behavior classifier. For example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

cv = LeaveOneGroupOut()
model = RandomForestClassifier()
cross_val_score(model, dataset.ml.features, dataset.ml.labels, groups = dataset.ml.group, cv = cv)
```

Since the `dataset` object is just an extended Pandas dataframe we can manipulate it as such. E.g. we can add our model predictions to the dataframe:
```python
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(model, dataset.ml.features, dataset.ml.labels, groups = dataset.ml.group, cv = cv)
dataset['prediction'] = predictions
```

If the raw video file paths are provided in the metadata, under the `video` key, we can make a movie overlaying these predictions over the original video
```python
dataset.io.save_movie(['label', 'prediction'], '.')
```

A more detailed run through of features is provided in the How To guide.

## References

[1] "The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice" Segalin et al, eLife 2021
