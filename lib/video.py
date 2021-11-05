""" Basic video class that houses data """

import pandas as pd
from glob import glob 
import pickle 
import numpy as np
from sklearn.model_selection import PredefinedSplit

class MLDataFrame(object):
    """
    DataFrame useful for interfacing between pandas and sklearn. Stores a data
    table and metadata dictionary. When feature columns, label columns and fold columns are specified
    then creates properties features, labels, folds and splitter that sklearn accepts for ML.
    """
    def __init__(self, data : pd.DataFrame,
                       metadata : dict = {},
                       fold_cols = None,
                       feature_cols = None,
                       label_cols = None):
        self.data = data 
        self.metadata = metadata
        self.fold_cols = fold_cols
        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def add_data(self, new_data, col_names):
        if type(new_data) == pd.DataFrame:
            self.data[col_names] = new_data
        else:
            self.data[col_names] = pd.DataFrame(new_data, index=self.data.index)

    @property
    def features(self):
        if self.feature_cols is None:
            return None
        else:
            return self.data[self.feature_cols].to_numpy()

    @property
    def labels(self):
        if self.label_cols is None:
            return None
        else:
            return self.data[self.label_cols].to_numpy()

    @property
    def folds(self):
        if self.fold_cols is None:
            return None
        else:
            return self.data[self.fold_cols].to_numpy()

    @property
    def splitter(self):
        if self.fold_cols is None:
            return None
        else:
            return self._make_predefined_split(self.folds)

    def _make_predefined_split(self, folds):
        test_indices = np.sum(folds, axis=1) == 0
        test_fold = -1*(test_indices)
        test_fold[test_indices == False] = np.argmin(folds[test_indices == False], axis = 1)
        return PredefinedSplit(test_fold)

    def save(self, fn):
        with open(fn, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        return str(self.data)

def clone_metadata(tracking_files, **kwargs):
    metadata = {}
    for fn in tracking_files:
        metadata[fn[0]] = kwargs

    return metadata

class VideosetDataFrame(MLDataFrame):
    def __init__(self, metadata : dict):
        self.req_cols = ['scale', 'fps', 'units', 'resolution']

        self.data = pd.DataFrame()
        self.label_key = None

        if self._validate_metadata(metadata):   
            self.metadata = metadata
        else:
            raise ValueError("Metadata not properly formatted. See docstring.")
    
        if len(metadata) > 0:
            self.load_tracks()
            self.load_labels()

    def _validate_metadata(self, metadata):
        for fn in metadata:
            checks = [col in fn.keys() for col in self.req_cols]
            if sum(checks) < len(self.req_cols):
                return False
        return True

    def load_tracks(self):
        raise NotImplementedError

    def load_labels(self):
        raise NotImplementedError

    def make_movie(self, fn_out, movie_in):
        raise NotImplementedError

 
def load_data(fn):
    try:
        with open(fn, 'rb') as handle:
            a = pickle.load(handle)
    except FileNotFoundError:
        print("Cannot find", fn)
        return None
    return a 

###############      
## Test code ##
###############

tracking_files = glob('./testdata/dlc/*.csv')
boris_files = glob('./testdata/boris/*.csv')

scale = None
fps = 30
units = None
resolution = (1200, 1600)
metadata = clone_metadata(tracking_files, 
                          label_file = boris_files, 
                          scale = scale, 
                          fps = fps, 
                          units = units, 
                          resolution = resolution)

dataset = VideosetDataFrame(metadata)

#Now create features on this dataset
dataset.create_dl_features()
dataset.create_mabe_features()
dataset.create_custom_features()
dataset.add_features()

#Set features by group names

#Now we can do ML on this object with the following attributes
dataset.features and dataset.label and dataset.splitter and/or dataset.group

"""
Advantages:
* Can import data from a range of sources
* Comes with some general data processing methods, e.g. can filter DLC tracks and interpolate at low-confidence points
* More general than SimBA and MABE
* Lightweight, no GUI... just use in jupyter notebook. Or can be put into a fully automated pipeline this way
 and be given to experimentalists. Train them to use DLC, BORIS and then run the script/notebook to do behavior classification
* For some problems (mouse tracking), good baseline performance 
* Extensible, add your own features;
* And try your own ML models or use good baselines
* Active learning... train classifier on one video, inference on the rest, and suggest chunks of 
"""