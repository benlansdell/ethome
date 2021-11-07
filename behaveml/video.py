""" Basic video tracking and behavior class that houses data """

import pandas as pd
from glob import glob 
import pickle 
import numpy as np
from sklearn.model_selection import PredefinedSplit

from behaveml.io import read_DLC_tracks

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

def _add_item_to_dict(tracking_files, metadata, k, item):
    for fn in tracking_files:
        metadata[fn][k] = item

def _add_items_to_dict(tracking_files, metadata, k, items):
    for fn, item in zip(tracking_files, items):
        metadata[fn][k] = item

def clone_metadata(tracking_files, **kwargs):
    """
    Prepare a metadata dictionary for defining a VideosetDataFrame. 

    Only required argument is list of DLC tracking file names. 

    Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS)
    that will be copied and tagged to each of the DLC tracking files, or an iterable object of the same
    length of the list of DLC tracking files. Each element in the iterable will be tagged with the corresponding
    DLC file.

    Args:
        tracking_files: (list) of DLC tracking .csvs
        **kwargs: described as above

    Returns:
        (dict) Dictionary whose keys are DLC tracking file names, and contains a dictionary with key,values containing
        the metadata provided
    """

    metadata = {}
    for fn in tracking_files:
        metadata[fn] = {}
    n_files = len(tracking_files)

    for k,v in kwargs.items():
        if hasattr(v, '__len__'):
            if len(v) != n_files:
                raise ValueError("Argument must be iterable same length as tracking_files")
            _add_items_to_dict(tracking_files, metadata, k, v)
        else:
            _add_item_to_dict(tracking_files, metadata, k, v)

    return metadata

class VideosetDataFrame(MLDataFrame):
    """
    Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata

    Args:
        metadata: (dict) Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata
            for that video. Most easiest to create with 'clone_metadata'. 
            Required keys are: ['scale', 'fps', 'units', 'resolution', 'label_files']
        label_key: (dict) Default None. Dictionary whose keys are behavior labels and values are integers 
    """
    def __init__(self, metadata : dict, label_key : dict = None):
        self.req_cols = ['scale', 'fps', 'units', 'resolution', 'label_files']

        self.data = pd.DataFrame()
        self.label_key = label_key
        self.reverse_label_key = {v:k for k,v in self.label_key.items()}

        if self._validate_metadata(metadata):   
            self.metadata = metadata
        else:
            raise ValueError("Metadata not properly formatted. See docstring.")
    
        self.n_videos = len(metadata)

        if len(metadata) > 0:
            self.load_tracks() 
            #By default make these tracks the features... 
            #useless for ML but just to have something to start with
            #This will be updated once some features to do ML with have been computed
            self.feature_cols = self.data.columns
            self.load_labels(set_as_label = True)

    def _setup_default_cv_folds(self):
        default_fold_cols = [f'fold{idx}' for idx in range(self.n_videos)]
        self.fold_cols = default_fold_cols 
        raise NotImplementedError

    def _validate_metadata(self, metadata):
        for fn in metadata:
            checks = [col in fn.keys() for col in self.req_cols]
            if sum(checks) < len(self.req_cols):
                return False
        return True

    def load_tracks(self):
        """Add DLC tracks to DataFrame"""
        raise NotImplementedError

    def load_labels(self, col_name = 'label', set_as_label = False):
        """Add behavior label data to DataFrame"""
        raise NotImplementedError

    def make_movie(self, prediction_column, fn_out, movie_in):
        """Given a column indicating behavior predictions, make a video
        outputting those predictiions alongside true labels."""
        raise NotImplementedError