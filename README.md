[![codecov](https://codecov.io/gh/benlansdell/ethome/branch/master/graph/badge.svg?token=IJ0JJBOGGS)](https://codecov.io/gh/benlansdell/ethome)
![Tests](https://github.com/benlansdell/ethome/actions/workflows/workflow.yml/badge.svg)

# Ethome

Machine learning for animal behavior.

Interprets pose-tracking files (currently only from DLC) and behavior annotations (currently only from BORIS) to train a behavior classifier, perform unsupervised learning, and other common analysis tasks. 

## Features

* Interpolate DLC data 
* Create generic features for kinematic analysis and downstream ML tasks
* Create features specifically for mouse resident-intruder setup
* Read in DLC pose data and corresponding BORIS behavior annotations to make supervised learning easy
* Perform unsupervised learning on pose data to extract discrete behavioral motifs (MotionMapper)
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
from glob import glob 
from ethome import createExperiment, clone_metadata
from ethome.features import CNN1DProb, MARS
from ethome.io import get_sample_data_paths
```

Gather the DLC and BORIS tracking and annotation files
```python
tracking_files, boris_files = get_sample_data_paths()
```

Setup some parameters
```python
frame_width = 20                 # (float) length of entire horizontal shot
frame_width_units = 'in'         # (str) units frame_width is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels
```

Create a parameter object and video dataset
```python
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

animal_renamer = {'adult': 'resident', 'juvenile':'intruder'}

dataset = createExperiment(metadata, animal_renamer=animal_renamer)
```

Now create features on this dataset. Feature creation objects are class instances, similar to sklearn:
```python
cnn_probabilities = CNN1DProb()
mars = MARS()

dataset.features.add(cnn_probabilities, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)

dataset.features.add(mars, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
```

Now access a features table, labels, and groups for learning with `dataset.ml.features, dataset.ml.labels, dataset.ml.groups`. From here it's easy to use some ML libraries to predict behavior. For example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

model = RandomForestClassifier()
cross_val_score(model, dataset.ml.features, dataset.ml.labels, dataset.ml.groups)
```

Read more in the how-to guide in the docs.
