[![codecov](https://codecov.io/gh/benlansdell/ethome/branch/master/graph/badge.svg?token=IJ0JJBOGGS)](https://codecov.io/gh/benlansdell/ethome)
![build](https://github.com/benlansdell/ethome/actions/workflows/workflow.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ethome-ml.svg)](https://badge.fury.io/py/ethome-ml)
[![status](https://joss.theoj.org/papers/0472dab158806827a83da79e602e16e4/status.svg)](https://joss.theoj.org/papers/0472dab158806827a83da79e602e16e4)

# Ethome

Tools for machine learning of animal behavior.

This library interprets pose-tracking files and behavior annotations to create features, train behavior classifiers, interpolate pose tracking data and other common analysis tasks.

At present pose tracking data from DLC, SLEAP and NWB formats are supported, and behavior annotations from BORIS and NWB formats are supported.

Full documentation is posted here: [https://benlansdell.github.io/ethome/](https://benlansdell.github.io/ethome/).

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

`ethome` has been tested with Python 3.7 and 3.8.

### Conda environment

Note that dependencies have tried to be kept to a minimum so that `ethome` can work easily alongside other programs that may be part of your behavior analysis pipeline (e.g. `DeepLabCut`) -- thus you can try running the `pip install` line above in an existing virtual environment.

That said, you may want a separate environment for running `ethome`. A conda environment can be created with the following steps:

1. Download the conda environment yaml file [ethome-conda.yaml](www.google.com)
2. (From the location you downloaded the yaml file) Create the environment: `conda env create -f ethome-conda.yaml`
3. Run `conda activate ethome`
3. And finally `pip install ethome-ml`

### Optional packages

With both install methods, you may want to also install `tensorflow` (at least version 2.0) if you want to use the CNN features for a resident-intruder setup.

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
```python
dataset.pose.body_parts
```

A key functionality of `ethome` is the ability to easily create features for machine learning. You can use pre-built featuresets or make your own. For instance:
```python
dataset.features.add('distances')
```
will compute all distances between all body parts (both between and within animals).

We can load pose data from DLC, and behavior annotation data from BORIS, provided we also provide a little metadata for context. E.g.:
```python
pose_files, behavior_files = get_sample_data_paths_dlcboris()
metadata = create_metadata(pose_files, labels = behavior_files, fps = 30)
dataset = create_dataset(metadata)
```

There are featuresets specifically tailored for social mice studies (the resident-intruder setup). For instance,
```python
dataset.features.add('cnn1d_prob')
```
Uses a pretrained CNN to output probabilities of 3 behaviors (attack, mount, social investigation). For this, you must have labeled your body parts in a certain way (refer to [How To](https://benlansdell.github.io/ethome/how-to/)). Other, more generic, feature creation functions are provided that work for any animal configuration.

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

A more detailed run through of features is provided in the How To guide. Also checkout `examples` for working demos to quickly see how things work.

## Supported input data formats

The following animal pose/behavior annotation data formats are supported.

### DeepLabCut

[Main project page](https://github.com/DeepLabCut/DeepLabCut)

From DLC documentation: The labels are stored in a MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

### BORIS

[Main project page](https://www.boris.unito.it/)

Behavioral Observation Research Interactive Software: BORIS is an easy-to-use event logging software for video/audio coding and live observations. It provides flexible and powerful behavior annotation from a set of videos. Data should be exported to a csv so that `ethome` can import behavior annotations.

### NeuroData Without Borders (NWB)

[Main project page](https://www.nwb.org/)

NWB is a data standard for neurophysiology, providing neuroscientists with a common standard to share, archive, use, and build analysis tools for neurophysiology data. NWB is designed to store a variety of neurophysiology data, including data from intracellular and extracellular electrophysiology experiments, data from optical physiology experiments, and tracking and stimulus data. It includes ability to store animal behavior data and pose data, through the ndx-pose extenstion ([here](https://github.com/rly/ndx-pose)). Nwb files must have a PoseEstimationSeries and/or PoseEstimation datatypes in it to be importable into `ethome`.

### SLEAP

[Main project page](https://sleap.ai/)

SLEAP is an open source deep-learning based framework for multi-animal pose tracking. It can be used to track any type or number of animals and includes an advanced labeling/training GUI for active learning and proofreading. SLEAP data must be in exported analysis `h5` files, to import into `ethome`.

## Sample notebooks

Sample notebooks are available ([here](https://github.com/benlansdell/ethome/tree/master/examples)) that you can use as a starting point for your own analyses, using either:
* NWB files
* SLEAP files
* DLC tracking and BORIS annotations

## Contributing

Refer to `CONTRIBUTING.md` for guidelines on how to contribute to the project, and report bugs, etc.

## Animal data

Sample data was obtained from resident-intruder open field recordings performed as part of on going social memory studies performed in the Zakharenko lab at St Jude Children's Research Hospital (e.g. [1,2]). All animal experiments were reviewed and approved by the Institutional Animal Care & Use Committee of St. Jude Children’s Research Hospital.

[1] "SCHIZOPHRENIA-RELATED MICRODELETION GENE 2510002D24Rik IS ESSENTIAL FOR SOCIAL MEMORY" US Patent US20220288235A1. Stanislav S. Zakharenko, Prakash DEVARAJU https://patents.google.com/patent/US20220288235A1/en
[2] "A murine model of hnRNPH2-related neurodevelopmental disorder reveals a mechanism for genetic compensation by Hnrnph1". Korff et al. Journal of clinical investigation 133(14). 2023.
