[![codecov](https://codecov.io/gh/benlansdell/ethome/branch/master/graph/badge.svg?token=IJ0JJBOGGS)](https://codecov.io/gh/benlansdell/ethome)
![tests](https://github.com/benlansdell/ethome/actions/workflows/workflow.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ethome-ml.svg)](https://badge.fury.io/py/ethome-ml)

# Ethome

Machine learning for animal behavior.

Interprets pose-tracking files (from DLC, NWB formats) and behavior annotations (from BORIS, NWB formats) to train a behavior classifier, perform unsupervised learning, and other common analysis tasks. 

## Features

* Read in DLC pose data and corresponding BORIS behavior annotations to make supervised learning easy
* Interpolate pose data to improve low-confidence predictions 
* Create generic features for kinematic analysis and downstream ML tasks
* Create features specifically for mouse resident-intruder setup
* Perform unsupervised learning on pose data to extract discrete behavioral motifs (similar to MotionMapper)
* Quickly generate a movie with behavior predictions

## Installation

```
pip install ethome-ml
```

Can install optional extras with:

```
pip install numpy, cython
pip install ethome-ml[all]
```

This includes matplotlib, keras, and Linderman lab's state-space model package, [ssm](https://github.com/lindermanlab/ssm). Note that installing ssm requires cython and numpy for the build, so must be already present in the environment. 

## Quickstart

Import
```python
from ethome import create_experiment, clone_metadata
from ethome.io import get_sample_data_paths
```

Gather the DLC tracking and BORIS annotation files
```python
tracking_files, boris_files = get_sample_data_paths()
```

Setup some parameters. All fields but `fps` are optional.
```python
frame_width = 20                 # (float) length of entire horizontal shot
frame_width_units = 'in'         # (str) units frame_width is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels
```

Create a parameter object and load the dataset
```python
metadata = clone_metadata(tracking_files, 
                          labels = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

animal_renamer = {'adult': 'resident', 'juvenile':'intruder'}

dataset = create_experiment(metadata, animal_renamer=animal_renamer)
```
`dataset` is an extended pandas DataFrame, so can be treated exactly as you would any other dataframe. But it adds for instance metadata about the pose:
```
dataset.pose.body_parts
```

And also it adds the ability to easily create features on this dataset. Can use pre-built featuresets, or make your own. Here are two that work with a mouse resident-intruder setup:
```python
dataset.features.add('cnn1d_prob')
dataset.features.add('mars')
```
Other, more generic feature creation functions are provided that work for any animal configuration.

(The 'mars' feature-set is designed for studying social behavior in mice, based heavily on Segalin et al. [1])

Now access a features table, labels, and groups for learning with `dataset.ml.features, dataset.ml.labels, dataset.ml.group`. From here it's easy to use some ML libraries to train a behavior classifier. For example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()
cross_val_score(model, dataset.ml.features, dataset.ml.labels, groups = dataset.ml.group)
```

Read more in the how-to guide in the docs.

## References

[1] "The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice" Segalin et al, eLife 2021
