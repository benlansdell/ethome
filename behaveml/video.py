""" Basic video tracking and behavior class that houses data. 

Basic object is the VideoSetDataFrame class.

## A note on unit conversions

For the unit rescaling, if the dlc/tracking file is already in desired units, either in physical distances, or pixels, then don't provide all of 'frame_width', 'resolution', and 'frame_width_units'. If you want to keep track of the units, you can add a 'units' key to the metadata. This could be 'pixels', or 'cm', as appropriate.

If the tracking is in pixels and you do want to rescale it to some physical distance, you should provide 'frame_width', 'frame_width_units' and 'resolution' for all videos. This ensures the entire dataset is using the same units. The package will use these values for each video to rescale the (presumed) pixel coordinates to physical coordinates. 

Resolution is a tuple (H,W) in pixels of the videos. 'frame_width' is the width of the image, in units 'frame_width_units'

When this is done, all coordinates are converted to 'mm'. The pair 'units':'mm' is added to the metadata dictionary for each video

If any of the provided parameters are provided, but are not the right format, or some values are missing, a warning is given and the rescaling is not performed.
"""

import pandas as pd
import pickle 
import os
import numpy as np
import re
from glob import glob
from sklearn.model_selection import PredefinedSplit

from behaveml.features import Features

from behaveml.io import read_DLC_tracks, read_boris_annotation, uniquifier, create_behavior_labels
from behaveml.utils import checkFFMPEG

from behaveml.config import global_config

#This converts everything to mm, or leaves them as pixels
UNIT_DICT = {'mm':1, 'cm':10, 'm':1000, 'in':25.4, 'ft':304.8,
             'millimeters':1, 'centimeters':10, 'meters':1000, 'inches':25.4, 'feet':304.8,
             'millimeter':1, 'centimeter':10, 'meter':1000, 'inch':25.4, 'foot':304.8,
             'millimetres':1, 'centimetres':10, 'metres':1000,
             'millimetre':1, 'centimetre':10, 'metre':1000}

class MLDataFrame(object): # pragma: no cover
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

def clone_metadata(tracking_files : list, **kwargs) -> dict:
    """
    Prepare a metadata dictionary for defining a VideosetDataFrame. 

    Only required argument is list of DLC tracking file names. 

    Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS)
    that will be copied and tagged to each of the DLC tracking files, or an iterable object of the same
    length of the list of DLC tracking files. Each element in the iterable will be tagged with the corresponding
    DLC file.

    Args:
        tracking_files: List of DLC tracking .csvs
        **kwargs: described as above

    Returns:
        Dictionary whose keys are DLC tracking file names, and contains a dictionary with key,values containing the metadata provided
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
    def __init__(self, metadata : dict, 
                       label_key : dict = None, 
                       part_renamer : dict = None,
                       animal_renamer : dict = None):
        """Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels.

        Args:
            metadata: Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata
                for that video. Most easiest to create with 'clone_metadata'. 
                Required keys are: ['fps']
            label_key: Default None. Dictionary whose keys are positive integers and values are behavior labels. If none, then this is inferred from the behavior annotation files provided.  
            part_renamer: Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.)
            animal_renamer: Default None. Dictionary that can rename animals from tracking files if needed
        """

        self.req_cols = ['fps']

        self.data = pd.DataFrame()
        self.label_key = label_key

        is_valid, should_rescale = self._validate_metadata(metadata)
        if is_valid:   
            self.metadata = metadata
        else:
            raise ValueError("Metadata not properly formatted. See docstring.")
    
        if len(metadata) > 0:
            self._load_tracks(part_renamer, animal_renamer, rescale = should_rescale) 
            self._load_labels(set_as_label = True)
        else:
            self.raw_track_columns = None

        if self.label_key:
            self.reverse_label_key = {v:k for k,v in self.label_key.items()}
        else: 
            self.reverse_label_key = None

        if should_rescale:
            self._convert_units()
        elif 'scale_factor' in self.data.columns: 
            self.data = self.data.drop(columns = 'scale_factor')

        self.feature_cols = None

    def _convert_units(self):
        # if 'frame_width', 'resolution' and 'frame_width_units' are provided, then we convert DLC tracks to these units.
        if len(self.metadata) == 0:
            return 
        for col in self.data.columns:
            is_dlc_feature = False
            for ani in self.animals:
                if col.startswith(ani):
                    is_dlc_feature = True
                    break
            if is_dlc_feature:
                self.data[col] = self.data[col]*self.data['scale_factor']
        self.data = self.data.drop(columns = 'scale_factor')

    def _setup_default_cv_folds(self):
        default_fold_cols = [f'fold{idx}' for idx in range(self.n_videos)]
        self.fold_cols = default_fold_cols 
        raise NotImplementedError

    def _validate_metadata(self, metadata):

        has_all_dim_cols_count = 0

        should_rescale = None
        valid = True

        import numbers
        import warnings

        for fn in metadata:

            n_dim_cols = sum([x in metadata[fn].keys() for x in ['frame_width', 'resolution', 'frame_width_units']])
            if (n_dim_cols > 0) and (n_dim_cols < 3):
                warnings.warn("Must provide all of 'frame_width', 'resolution' and 'frame_width_units' to rescale")
                should_rescale = False
            if (n_dim_cols == 3):
                has_all_dim_cols_count += 1
                if should_rescale is not False:
                    should_rescale = True

                if not isinstance(metadata[fn]['frame_width'], numbers.Number):
                    warnings.warn("'frame_width' must be a number to rescale.")
                    should_rescale = False
                if type(metadata[fn]['frame_width_units']) is not str:
                    warnings.warn("'frame_width_units' must be a string to rescale.")
                    should_rescale = False
                elif metadata[fn]['frame_width_units'].lower() not in UNIT_DICT:
                    warnings.warn(f"Units must be one of the following to rescale: {','.join(UNIT_DICT.keys())}")
                    should_rescale = False

                if hasattr(metadata[fn]['resolution'], '__len__'):
                    if len(metadata[fn]['resolution']) != 2:
                        warnings.warn("'resolution' must be a list-like object of length 2 to rescale.")
                        should_rescale = False

            checks = [col in metadata[fn].keys() for col in self.req_cols]
            if sum(checks) < len(self.req_cols):
                valid = False

        if (has_all_dim_cols_count > 0) and (has_all_dim_cols_count < len(metadata)):
            warnings.warn("Must provide all of 'frame_width', 'resolution' and 'frame_width_units' to rescale")
            should_rescale = False

        if should_rescale is None: should_rescale = False
        if should_rescale:
            print("Rescaling to 'mm'")

        return valid, should_rescale

    @property
    def videos(self):
        return list(self.metadata.keys())

    @property
    def n_videos(self):
        return len(self.metadata)

    @property
    def group(self):
        return self.data.filename.to_numpy()

    def activate_features_by_name(self, name : str) -> list:
        """Add already present columns in data frame to the feature set. 
        
        Args:
            name: string for pattern matching -- any feature that starts with this string will be added

        Returns:
            List of matched columns (may include columns that were already activated).
        """

        matched_cols = [l for l in self.data.columns if re.match(f"^{name}", l)]
        if self.feature_cols is not None:
            self.feature_cols = uniquifier(list(self.feature_cols) + list(matched_cols))
        else:
            self.feature_cols = matched_cols
        return matched_cols

    def get_columns_regex(self, pattern : str) -> list:
        """Return a list of column names that match the provided regex pattern.
        
        Args:
            pattern: a regex pattern to match column names to
            
        Returns:
            list of column names
        """
        try:
            compiled = re.compile(pattern)
        except re.error:
            raise ValueError("Couldn't parse re pattern.")
        matched_cols = [l for l in self.data.columns if compiled.search(l) is not None]
        return matched_cols

    def remove_features_by_name(self, name : str) -> list:
        """Remove columns from the feature set. 
        
        Args:
            name: string for pattern matching -- any feature that starts with this string will be removed

        Returns:
            List of removed columns.
        """
        matched_cols = [l for l in self.feature_cols if re.match(f"^{name}", l)]
        removed_cols = self.remove_feature_cols(matched_cols)
        return removed_cols

    #Set features by individual or by group names
    def add_features(self, feature_maker : Features, 
                           featureset_name : str, 
                           add_to_features = False, 
                           **kwargs) -> list:
        """Compute features to dataframe using Feature object. 'featureset_name' will be prepended to new columns, followed by a double underscore. 

        Args:
            featuremaker: A Feature object that houses the feature-making function to be executed and a list of required columns that must in the dataframe for this to work
            featureset_name: Name to prepend to the added features 
            add_to_features: Whether to add to list of active features (i.e. will be returned by the .features property)
        Returns:
            List of new columns that are computed
        """
        new_features = feature_maker.make(self, **kwargs)

        #Prepend these column names w featureset-name__feature-name
        new_feat_cols = list(new_features.columns)
        new_feat_cols = [str(featureset_name) + '__' + str(i) for i in new_feat_cols]
        new_features.columns = new_feat_cols

        #Don't add duplicated columns:
        notdupcols = [col for col in new_feat_cols if col not in self.data.columns]
        new_features = new_features[notdupcols]

        self.data = pd.concat([self.data.reset_index(drop = True), 
                               new_features.reset_index(drop = True)], axis = 1)
        if add_to_features:
            if self.feature_cols is not None:
                self.feature_cols = list(self.feature_cols) + list(new_features.columns)
            else:
                self.feature_cols = new_features.columns
        return list(new_features.columns)

    def remove_feature_cols(self, col_names : list) -> list:
        """Remove provided columns from set of feature columns.
        
        Args:
            col_names: list of column names
            
        Returns:
            The columns that were removed from those designated as features."""
        new_col_names = [i for i in self.feature_cols if i not in col_names]
        removed = [i for i in self.feature_cols if i in col_names]
        self.feature_cols = new_col_names
        return removed

    def _load_tracks(self, part_renamer, animal_renamer, rescale = False):
        #For the moment only supports DLC
        return self._load_dlc_tracks(part_renamer, animal_renamer, rescale = rescale)

    def _load_dlc_tracks(self, part_renamer, animal_renamer, rescale = False):
        """Add DLC tracks to DataFrame"""
        df = pd.DataFrame()
        dfs = []
        col_names_old = None
        #Go through each video file and load DLC tracks
        for fn in self.metadata.keys():
            df_fn, body_parts, animals, col_names, scorer = read_DLC_tracks(fn, 
                                                                            part_renamer, 
                                                                            animal_renamer)
            n_rows = len(df_fn)
            df_fn['time'] = df_fn['frame']/self.metadata[fn]['fps']
    
            if rescale and ('frame_width_units' in self.metadata[fn]) and \
                    ('frame_width' in self.metadata[fn]) and \
                    ('resolution' in self.metadata[fn]) and \
                    self.metadata[fn]['frame_width_units'].lower() in UNIT_DICT:
                unit_scale_factor = UNIT_DICT[self.metadata[fn]['frame_width_units'].lower()]
                df_fn['scale_factor'] = self.metadata[fn]['frame_width']/self.metadata[fn]['resolution'][1]*unit_scale_factor
                self.metadata[fn]['units'] = 'mm'
            else:
                df_fn['scale_factor'] = 1

            dfs.append(df_fn)
            self.metadata[fn]['duration'] = (n_rows)/self.metadata[fn]['fps']
            self.metadata[fn]['scorer'] = scorer
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
        for animal in self.animals:
            if animal in ['filename', 'frame', 'time', 'label']:
                print(f"One of the animal names is protected ({animal}), unexpected behavior may occur.")

    def _load_labels(self, col_name = 'label', set_as_label = False):
        #For the moment only BORIS support
        return self._load_labels_boris(col_name, set_as_label)

    def _load_labels_boris(self, col_name = 'label', set_as_label = False):
        """Add behavior label data to DataFrame"""

        if self.label_key is None:
            label_files = []
            for fn in self.metadata.keys():
                if 'label_files' in self.metadata[fn]:
                    label_files.append(self.metadata[fn]['label_files'])
            self.label_key = create_behavior_labels(label_files)

        for vid in self.metadata:
            if 'label_files' in self.metadata[vid]:

                fn_in = self.metadata[vid]['label_files']
                fps = self.metadata[vid]['fps']
                duration = self.metadata[vid]['duration']

                ground_truth, _ = read_boris_annotation(fn_in, fps, duration, self.label_key)

                self.data.loc[self.data['filename'] == vid, col_name] = ground_truth

        if set_as_label:
            self.label_cols = col_name

    def save(self, fn_out : str) -> None:
        """Save VideosetDataFrame object with pickle.
        
        Args:
            fn_out: location to write pickle file to
            
        Returns:
            None. File is saved to path.
        """
        with open(fn_out,'wb') as file:
            file.write(pickle.dumps(self.__dict__, protocol = 4))

    def to_dlc_csv(self, base_dir : str, save_h5_too = False) -> None:
        """Save VideosetDataFrame tracking files to DLC csv format.

        Only save tracking data, not other computed features.
        
        Args:
            base_dir: base_dir to write DLC csv files to
            save_h5_too: if True, also save the data as an h5 file
            
        Returns:
            None. Files are saved to path.
        """
        from itertools import product
        fns = list(self.metadata.keys())
        for fn in fns:
            df = self.data.loc[self.data['filename'] == fn].copy().reset_index(drop = True)
            #Make a multi column object and rearrange table to be in DLC format
            scorer = self.metadata[fn]['scorer']
            selected_cols = []
            for animal, bp, component in product(self.animals, self.body_parts, ['x', 'y', 'likelihood']):
                if component == 'likelihood':
                    old_col_name = f"{component}_{animal}_{bp}"
                else:
                    old_col_name = f"{animal}_{component}_{bp}"
                new_col_name = (scorer, animal, bp, component)
                df = df.rename(columns = {old_col_name: new_col_name})
                selected_cols.append(new_col_name)
            df = df[selected_cols]
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=['scorer', 'individuals', 'bodyparts', 'coords'])
            fn_out = os.path.join(base_dir, os.path.basename(fn))
            df.to_csv(fn_out)
            if save_h5_too:
                df.to_hdf(fn_out.replace('.csv', '.h5'), "df_with_missing", format = 'table', mode="w")

    def load(self, fn_in : str) -> None:
        """Load VideosetDataFrame object from pickle file.
        
        Args:
            fn_in: path to load pickle file from. 
            
        Returns:
            None. Data in this object is populated with contents of file."""
        with open(fn_in, 'rb') as file:
            dataPickle = file.read()
        self.__dict__ = pickle.loads(dataPickle)

    def make_movie(self, label_columns, path_out : str, video_filenames = None) -> None:
        """Given columns indicating behavior predictions or whatever else, make a video
        with these predictions overlaid. 

        VideosetDataFrame metadata must have the keys 'video_file', so that the video associated with each set of DLC tracks is known.

        Args:
            label_columns: list or dict of columns whose values to overlay on top of video. If dict, keys are the columns and values are the print-friendly version.
            path_out: the directory to output the videos too
            video_filenames: list or string. The set of videos to use. If not provided, then use all videos as given in the metadata.

        Returns:
            None. Videos are saved to 'path_out'
        """

        if not checkFFMPEG():
            print("Cannot find ffmpeg in path. Please install.")
            return

        #Text parameters (units in pixels)
        y_offset = global_config['make_movie__y_offset']
        y_inc = global_config['make_movie__y_inc']
        text_color = global_config['make_movie__text_color']
        font_size = global_config['make_movie__font_size']

        if type(video_filenames) is str:
            video_filenames = [video_filenames]
        if not video_filenames:
            video_filenames = self.videos
        if type(label_columns) is list:
            label_columns = {k:k for k in label_columns}

        for video in video_filenames:
            rate = 1/self.metadata[video]['fps']
            vid_in = self.metadata[video]['video_files']
            file_out = os.path.splitext(os.path.basename(vid_in))[0] + '_'.join(label_columns.keys()) + '.mp4'
            vid_out = os.path.join(path_out, file_out)
            label_strings = []
            for idx, (col,print_label) in enumerate(label_columns.items()):
                this_y_offset = y_offset + y_inc*idx
                behavior_pairs = _make_dense_values_into_pairs(self.data.loc[self.data.filename == video, col], rate)
                this_label_string = ','.join([f"drawtext=text=\'{print_label}\':x=90:y={this_y_offset}:fontsize={font_size}:fontcolor={text_color}:enable=\'between(t,{pair[0]},{pair[1]})\'" for pair in behavior_pairs])
                label_strings.append(this_label_string)
            label_string = ','.join(label_strings)

            #ffmpeg is the fastest way to add this information to a video
            #Prepare the ffmpeg command
            cmd = f'ffmpeg -y -i {vid_in} -vf "{label_string}" {vid_out}'
            os.system(cmd)
            
def load_videodataset(fn_in : str) -> VideosetDataFrame:
    """Load VideosetDataFrame from file.
    
    Args:
        fn_in: path to file to load
        
    Returns:
        VideosetDataFrame object from pickle file
    """
    with open(fn_in, 'rb') as file:
        dataPickle = file.read()
    new_obj = VideosetDataFrame({})
    new_obj.__dict__ = pickle.loads(dataPickle)
    return new_obj

def get_sample_openfield_data():
    """Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC.
    
    Returns:
        (VideosetDataFrame) Data frame with the corresponding tracking and behavior annotation files
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, 'data', 'dlc', 'openfield', '*example.csv'))
    video_files = glob(os.path.join(cur_dir, 'data', 'videos', '*.mp4'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)        # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = clone_metadata(tracking_files, 
                            video_files = video_files,
                            fps = fps, 
                            resolution = resolution)

    vdf = VideosetDataFrame(metadata)

    return vdf

def _make_dense_values_into_pairs(predictions, rate):
    #Put into start/stop pairs
    pairs = []
    in_pair = False
    start = 0
    for idx, behavior in enumerate(predictions):
        if behavior == 0:
            if in_pair:
                pairs.append([start, idx*rate])
            in_pair = False
        if behavior == 1:
            if not in_pair:
                start = idx*rate
            in_pair = True

    if in_pair:
        pairs.append([start, idx*rate])
    return pairs
    