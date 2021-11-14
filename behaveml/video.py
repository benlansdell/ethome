""" Basic video tracking and behavior class that houses data """

import pandas as pd
import pickle 
import numpy as np
from itertools import product
from glob import glob 
from sklearn.model_selection import PredefinedSplit

from behaveml.io import read_DLC_tracks, XY_IDS, rename_df_cols

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
                _add_item_to_dict(tracking_files, metadata, k, v)
            else:
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
        part_renamer: (dict) Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.)
        animal_renamer: (dict) Default None. Dictionary that re rename animals from tracking files if needed
    """
    def __init__(self, metadata : dict, 
                       label_key : dict = None, 
                       part_renamer : dict = None,
                       animal_renamer : dict = None):
        self.req_cols = ['frame_length', 'fps', 'units', 'resolution']

        self.data = pd.DataFrame()
        self.label_key = label_key
        if self.label_key:
            self.reverse_label_key = {v:k for k,v in self.label_key.items()}
        else: 
            self.reverse_label_key = None

        if self._validate_metadata(metadata):   
            self.metadata = metadata
        else:
            raise ValueError("Metadata not properly formatted. See docstring.")
    
        if len(metadata) > 0:
            self._load_tracks(part_renamer, animal_renamer) 
            #By default make these tracks the features... 
            #useless for ML but just to have something to start with
            #This will be updated once some features to do ML with have been computed
            self._load_labels(set_as_label = True)
        else:
            self.raw_track_columns = None

        self.feature_cols = None

    def _setup_default_cv_folds(self):
        default_fold_cols = [f'fold{idx}' for idx in range(self.n_videos)]
        self.fold_cols = default_fold_cols 
        raise NotImplementedError

    def _validate_metadata(self, metadata):
        for fn in metadata:
            checks = [col in metadata[fn].keys() for col in self.req_cols]
            if sum(checks) < len(self.req_cols):
                return False
        return True

    @property
    def videos(self):
        return list(self.metadata.keys())

    @property
    def n_videos(self):
        return len(self.metadata)

    @property
    def group(self):
        return self.data.filename.to_numpy()

    #Set features by individual or by group names
    def add_features(self, feature_maker, featureset_name, columns = None, add_to_features = False, **kwargs):
        """
        Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata

        Args:
            featuremaker: (dict) Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata
                for that video. Most easiest to create with 'clone_metadata'. 
                Required keys are: ['scale', 'fps', 'units', 'resolution', 'label_files']
            label_key: (dict) Default None. Dictionary whose keys are behavior labels and values are integers 
            part_renamer: (dict) Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.)
        Returns:
            None
        """
        if columns is None:
            columns = self.raw_track_columns
        new_features = feature_maker(self.data, columns, self.animal_setup, **kwargs)

        #Prepend these column names w featureset-name__feature-name
        new_feat_cols = list(new_features.columns)
        new_feat_cols = [str(featureset_name) + '__' + str(i) for i in new_feat_cols]
        new_features.columns = new_feat_cols

        self.data = pd.concat([self.data.reset_index(drop = True), 
                               new_features.reset_index(drop = True)], axis = 1)
        if add_to_features:
            if self.feature_cols:
                self.feature_cols = list(self.feature_cols) + list(new_features.columns)
            else:
                self.feature_cols = new_features.columns

    def remove_feature_cols(self, col_names):
        new_col_names = [i for i in self.feature_cols if i not in col_names]
        self.feature_cols = new_col_names

    def _load_tracks(self, part_renamer, animal_renamer):
        #For the moment only supports DLC
        return self._load_dlc_tracks(part_renamer, animal_renamer)

    def _load_dlc_tracks(self, part_renamer, animal_renamer):
        """Add DLC tracks to DataFrame"""
        df = pd.DataFrame()
        dfs = []
        col_names_old = None
        #Go through each video file and load DLC tracks
        for fn in self.metadata.keys():
            df_fn, body_parts, animals, col_names = read_DLC_tracks(fn, 
                                                                    part_renamer, 
                                                                    animal_renamer)
            n_rows = len(df_fn)
            dfs.append(df_fn)
            self.metadata[fn]['duration'] = n_rows/self.metadata[fn]['fps']
            if col_names_old is not None:
                if col_names != col_names_old:
                    raise RuntimeError("DLC files have different columns. Must all be from same project")
            col_names_old = col_names
        df = pd.concat(dfs, axis = 0)
        df = df.reset_index(drop = True)
        self.body_parts = body_parts
        self.animals = animals
        self.animal_setup = {'mouse_ids': animals, 'bodypart_ids': body_parts, 'colnames': col_names}
        self.raw_track_columns = col_names
        self.data = df

    def _load_labels(self, col_name = 'label', set_as_label = False):
        #For the moment only BORIS support
        return self._load_labels_boris(col_name, set_as_label)

    def _load_labels_boris(self, col_name = 'label', set_as_label = False):
        """Add behavior label data to DataFrame"""

        for vid in self.metadata:
            if 'label_files' in self.metadata[vid]:
                fn_in = self.metadata[vid]['label_files']
                fps = self.metadata[vid]['fps']
                n_bins = int(self.metadata[vid]['duration']*fps)
                boris_labels = pd.read_csv(fn_in, skiprows = 15)
                boris_labels['index'] = (boris_labels.index//2)
                boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
                boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
                boris_labels = [list(i) for i in boris_labels]
                ground_truth = np.zeros(n_bins)
                for start, end in boris_labels:
                    ground_truth[int(start*fps):int(end*fps)] = 1
                #Add this to data
                self.data.loc[self.data['filename'] == vid, col_name] = ground_truth

        if set_as_label:
            self.label_cols = col_name

    def make_movie(self, prediction_column, fn_out, movie_in):
        """Given a column indicating behavior predictions, make a video
        outputting those predictiions alongside true labels."""
        raise NotImplementedError