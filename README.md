[![codecov](https://codecov.io/gh/benlansdell/behaveml/branch/master/graph/badge.svg?token=PN52Q3UH3G)](https://codecov.io/gh/benlansdell/behaveml)

# BehaveML

Machine learning for animal behavior.

Interprets pose-tracking files (currently only from DLC), behavior annotations (currently only from BORIS) to train a behavior classifier. 

## Installation

```
pip install behaveml
```

Can install optional extras with:

```
pip install numpy, cython
pip install behaveml[all]
```

This includes matplotlib, keras, and Linderman lab's state-space model package, ssm. Note that installing ssm requires cython and numpy for the build, so must be already present in the environment. 

## Quickstart

Import
```
from glob import glob 
from behaveml import VideosetDataFrame, clone_metadata
from behaveml import compute_dl_probability_features, compute_mars_features
```

Gather the DLC and BORIS tracking and annotation files
```
tracking_files = glob('./tests/data/dlc/*.csv')
boris_files = glob('./tests/data/boris/*.csv')
```

Setup some parameters
```
frame_width = 20                 # (float) length of entire horizontal shot
frame_width_units = 'in'         # (str) units frame_width is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels
```

Create a parameter object and video dataset
```
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

animal_renamer = {'adult': 'resident', 'juvenile':'intruder'}

dataset = VideosetDataFrame(metadata, animal_renamer=animal_renamer)
```

Now create features on this dataset
```
dataset.add_features(compute_dl_probability_features, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)

dataset.add_features(compute_mars_features, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
```

Now access a features table, labels, and groups for learning with `dataset.features, dataset.labels, dataset.groups`.
For example:
```
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
predictions = cross_val_predict(model, dataset.features, dataset.labels, dataset.groups)
score = accuracy_score(dataset.labels, predictions)
```
