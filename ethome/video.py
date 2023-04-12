""" Basic video tracking and behavior class that houses data"""

import pandas as pd
import dill
import os
import numpy as np
import re
import warnings
import types 

from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, PredefinedSplit

from ethome.io import read_DLC_tracks, read_boris_annotation, uniquifier, create_behavior_labels, read_NWB_tracks, read_sleap_tracks
from ethome.utils import checkFFMPEG
from ethome.config import global_config
from ethome.features import feature_class_maker, FEATURE_MAKERS

#This converts everything to mm, or leaves them as pixels
UNIT_DICT = {'mm':1, 'cm':10, 'm':1000, 'in':25.4, 'ft':304.8,
             'millimeters':1, 'centimeters':10, 'meters':1000, 'inches':25.4, 'feet':304.8,
             'millimeter':1, 'centimeter':10, 'meter':1000, 'inch':25.4, 'foot':304.8,
             'millimetres':1, 'centimetres':10, 'metres':1000,
             'millimetre':1, 'centimetre':10, 'metre':1000}

def _add_item_to_dict(tracking_files, metadata, k, item):
    for fn in tracking_files:
        metadata[fn][k] = item

def _add_items_to_dict(tracking_files, metadata, k, items):
    for fn, item in zip(tracking_files, items):
        metadata[fn][k] = item

def create_metadata(tracking_files : list, **kwargs) -> dict:
    """
    Prepare a metadata dictionary for defining a ExperimentDataFrame. 

    Only required argument is list of pose tracking file names. 

    Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS)
    that will be copied and tagged to each of the pose tracking files, or an iterable object of the same
    length of the list of pose tracking files. Each element in the iterable will be tagged with the corresponding
    pose-tracking file.

    Args:
        tracking_files: List of pose tracking files
        **kwargs: described as above

    Returns:
        Dictionary whose keys are pose-tracking file names, and contains a dictionary with key,values containing the metadata provided
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

def _convert_units(df):
    # if 'frame_width', 'resolution' and 'frame_width_units' are provided, then we convert tracks to these units.
    if len(df.metadata.details) == 0:
        return 
    for col in df.columns:
        is_dlc_feature = False
        for ani in df.pose.animals:
            if col.startswith(ani):
                is_dlc_feature = True
                break
        if is_dlc_feature:
            df[col] = df[col]*df['scale_factor']
    df.drop(columns = 'scale_factor', inplace = True)

def _validate_metadata(metadata, req_cols):

    has_all_dim_cols_count = 0

    should_rescale = None
    valid = True

    import numbers
    import warnings

    from collections import defaultdict
    unit_counts = defaultdict(int)

    for fn in metadata:

        n_dim_cols = sum([x in metadata[fn].keys() for x in ['frame_width', 'resolution', 'frame_width_units']])
        if (n_dim_cols > 0) and (n_dim_cols < 3):
            warnings.warn("Must provide all of 'frame_width', 'resolution' and 'frame_width_units' to rescale. Not rescaling.")
            should_rescale = False
        if (n_dim_cols == 3):
            has_all_dim_cols_count += 1
            if should_rescale is not False:
                should_rescale = True

            if not isinstance(metadata[fn]['frame_width'], numbers.Number):
                warnings.warn("'frame_width' must be a number to rescale. Not rescaling.")
                should_rescale = False
            if type(metadata[fn]['frame_width_units']) is not str:
                warnings.warn("'frame_width_units' must be a string to rescale. Not rescaling.")
                should_rescale = False
            elif metadata[fn]['frame_width_units'].lower() not in UNIT_DICT:
                warnings.warn(f"Units must be one of the following to rescale: {','.join(UNIT_DICT.keys())}. Not rescaling.")
                should_rescale = False

            if hasattr(metadata[fn]['resolution'], '__len__'):
                if len(metadata[fn]['resolution']) != 2:
                    warnings.warn("'resolution' must be a list-like object of length 2 to rescale. Not rescaling.")
                    should_rescale = False
            else:
                warnings.warn("'resolution' must be a list-like object of length 2 to rescale. Not rescaling.")
                should_rescale = False

        if 'units' in metadata[fn].keys():
            unit_counts[metadata[fn]['units']] += 1
        else:
            unit_counts[0] += 1

        checks = [col in metadata[fn].keys() for col in req_cols]
        if sum(checks) < len(req_cols):
            valid = False

    if (has_all_dim_cols_count > 0) and (has_all_dim_cols_count < len(metadata)):
        warnings.warn("Must provide all of 'frame_width', 'resolution' and 'frame_width_units' to rescale. Not rescaling.")
        should_rescale = False

    if should_rescale is None: should_rescale = False

    for unit in unit_counts:
        if unit_counts[unit] > 0:
            if unit_counts[unit] != len(metadata):
                warnings.warn(f"Target units must be the same for all files. Not rescaling.")
                should_rescale = False

    if 0 in unit_counts:
        del unit_counts[0]
    if len(unit_counts) > 0:
        target_units = list(unit_counts.keys())[0]
    else:
        target_units = 'mm'

    if target_units not in UNIT_DICT:
        warnings.warn(f"Units must be one of the following: {','.join(UNIT_DICT.keys())}. Not rescaling.")
        should_rescale = False

    if should_rescale:
        print(f"All necessary fields provided -- rescaling to '{target_units}'")

    return valid, should_rescale

@pd.api.extensions.register_dataframe_accessor("metadata")
class EthologyMetadataAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        if 'metadata__details' not in self._obj.attrs:
            self._obj.attrs['metadata__details'] = {}
        if 'metadata__label_key' not in self._obj.attrs:
            self._obj.attrs['metadata__label_key'] = {}

    @property
    def details(self):
        return self._obj.attrs['metadata__details']

    @property
    def label_key(self):
        return self._obj.attrs['metadata__label_key']

    @details.setter
    def details(self, val):
        self._obj.attrs['metadata__details'] = val

    @label_key.setter
    def label_key(self, val):
        self._obj.attrs['metadata__label_key'] = val

    @property
    def videos(self):
        return list(self.details.keys())

    @property
    def n_videos(self):
        return len(self.videos)

    @property
    def reverse_label_key(self):
        return {v:k for k,v in self.label_key.items()}

@pd.api.extensions.register_dataframe_accessor("features")
class EthologyFeaturesAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        if 'features__active' not in self._obj.attrs:
            self._obj.attrs['features__active'] = None

    @property
    def active(self):
        return self._obj.attrs['features__active']

    @active.setter
    def active(self, val):
        self._obj.attrs['features__active'] = val

    def activate(self, name : str) -> list:
        """Add already present columns in data frame to the feature set. 
        
        Args:
            name: string for pattern matching -- any feature that starts with this string will be added

        Returns:
            List of matched columns (may include columns that were already activated).
        """

        matched_cols = [l for l in self._obj.columns if re.match(f"^{name}", l)]
        if self.active is not None:
            self.active = uniquifier(list(self.active) + list(matched_cols))
        else:
            self.active = matched_cols
        return matched_cols

    def regex(self, pattern : str) -> list:
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
        matched_cols = [l for l in self._obj.columns if compiled.search(l) is not None]
        return matched_cols

    def deactivate(self, name : str) -> list:
        """Remove columns from the feature set. 
        
        Args:
            name: string for pattern matching -- any feature that starts with this string will be removed

        Returns:
            List of removed columns.
        """
        matched_cols = [l for l in self.active if re.match(f"^{name}", l)]
        removed_cols = self.deactivate_cols(matched_cols)
        return removed_cols

    #Set features by individual or by group names
    def add(self, feature_maker, 
                           featureset_name : str = None, 
                           add_to_features = True, 
                           required_columns = [],
                           **kwargs) -> list:
        """Compute features to dataframe using Feature object. 'featureset_name' will be prepended to new columns, followed by a double underscore. 

        Args:
            featuremaker: A Feature object that houses the feature-making function to be executed and a list of required columns that must in the dataframe for this to work
            featureset_name: Name to prepend to the added features 
            add_to_features: Whether to add to list of active features (i.e. will be returned by the .features property)

        Returns:
            List of new columns that are computed
        """
        df = self._obj

        if isinstance(feature_maker, str):
            if feature_maker in FEATURE_MAKERS:
                FeatureMaker = FEATURE_MAKERS[feature_maker]
                if len(required_columns):
                    feature_maker = FeatureMaker(required_columns, **kwargs)
                else:
                    feature_maker = FeatureMaker(**kwargs)
            else:
                raise ValueError(f"Feature maker {feature_maker} not found.")

        if isinstance(feature_maker, types.FunctionType):
            CustomFeatureClass = feature_class_maker('CustomFeatureClass', feature_maker, required_columns)
            feature_maker = CustomFeatureClass()

        if featureset_name is None:
            try:
                featureset_name = feature_maker.__name__
            except AttributeError:
                featureset_name = feature_maker.__class__.__name__

        new_features = feature_maker.transform(df, **kwargs)

        #Prepend these column names w featureset-name__feature-name
        new_feat_cols = list(new_features.columns)
        new_feat_cols = [str(featureset_name) + '__' + str(i) for i in new_feat_cols]
        new_features.columns = new_feat_cols

        #Don't add duplicated columns:
        notdupcols = [col for col in new_feat_cols if col not in df.columns]
        new_features = new_features[notdupcols]

        df.reset_index(drop = True, inplace = True)
        #Suppress warnings here:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[notdupcols] = new_features.reset_index(drop = True)

        if add_to_features:
            if self.active is not None:
                self.active = list(self.active) + list(new_features.columns)
            else:
                self.active = new_features.columns
        return list(new_features.columns)

    def deactivate_cols(self, col_names : list) -> list:
        """Remove provided columns from set of feature columns.
        
        Args:
            col_names: list of column names
            
        Returns:
            The columns that were removed from those designated as features."""
        new_col_names = [i for i in self.active if i not in col_names]
        removed = [i for i in self.active if i in col_names]
        self.active = new_col_names
        return removed


@pd.api.extensions.register_dataframe_accessor("pose")
class EthologyPoseAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

        if 'pose__body_parts' not in self._obj.attrs:
            self._obj.attrs['pose__body_parts'] = None

        if 'pose__animals' not in self._obj.attrs:
            self._obj.attrs['pose__animals'] = None

        if 'pose__animal_setup' not in self._obj.attrs:
            self._obj.attrs['pose__animal_setup'] = None

        if 'pose__raw_track_columns' not in self._obj.attrs:
            self._obj.attrs['pose__raw_track_columns'] = None

    @property
    def body_parts(self):
        return self._obj.attrs['pose__body_parts']

    @body_parts.setter
    def body_parts(self, val):
        self._obj.attrs['pose__body_parts'] = val

    @property
    def animals(self):
        return self._obj.attrs['pose__animals']

    @animals.setter
    def animals(self, val):
        self._obj.attrs['pose__animals'] = val

    @property
    def animal_setup(self):
        return self._obj.attrs['pose__animal_setup']

    @animal_setup.setter
    def animal_setup(self, val):
        self._obj.attrs['pose__animal_setup'] = val

    @property
    def raw_track_columns(self):
        return self._obj.attrs['pose__raw_track_columns']

    @raw_track_columns.setter
    def raw_track_columns(self, val):
        self._obj.attrs['pose__raw_track_columns'] = val

@pd.api.extensions.register_dataframe_accessor("ml")
class EthologyMLAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

        if 'ml__label_cols' not in self._obj.attrs:
            self._obj.attrs['ml__label_cols'] = None

        if 'ml__fold_cols' not in self._obj.attrs:
            self._obj.attrs['ml__fold_cols'] = None

    @property
    def label_cols(self):
        return self._obj.attrs['ml__label_cols']

    @label_cols.setter
    def label_cols(self, val):
        self._obj.attrs['ml__label_cols'] = val

    @property
    def fold_cols(self):
        return self._obj.attrs['ml__fold_cols']

    @fold_cols.setter
    def fold_cols(self, val):
        self._obj.attrs['ml__fold_cols'] = val

    @property
    def features(self):
        df = self._obj
        if df.features.active is not None:
            return df[df.features.active].to_numpy()
        else:
            return None

    @property
    def labels(self):
        if not self.label_cols:
            return None
        else:
            return self._obj[self.label_cols].to_numpy()

    @property
    def folds(self): # pragma: no cover
        if not self.fold_cols:
            return None
        else:
            return self._obj[self.fold_cols].to_numpy()

    @property
    def group(self):
        return self._obj['filename'].to_numpy()

    @property
    def splitter(self): # pragma: no cover
        if not self.fold_cols:
            return None
        else:
            return self._make_predefined_split(self.folds)

    def _make_predefined_split(self, folds): # pragma: no cover
        test_indices = np.sum(folds, axis=1) == 0
        test_fold = -1*(test_indices)
        test_fold[test_indices == False] = np.argmin(folds[test_indices == False], axis = 1)
        return PredefinedSplit(test_fold)

@pd.api.extensions.register_dataframe_accessor("io")
class EthologyIOAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def save(self, fn_out : str) -> None:
        """Save ExperimentDataFrame object with pickle.
        
        Args:
            fn_out: location to write pickle file to
            
        Returns:
            None. File is saved to path.
        """
        df = self._obj
        with open(fn_out,'wb') as file:
            #file.write(pickle.dumps(df.__dict__, protocol = 4))
            file.write(dill.dumps(df.__dict__, protocol = 4))

    def to_dlc_csv(self, base_dir : str, save_h5_too = False) -> None:
        """Save ExperimentDataFrame tracking files to DLC csv format.

        Only save tracking data, not other computed features.
        
        Args:
            base_dir: base_dir to write DLC csv files to
            save_h5_too: if True, also save the data as an h5 file
            
        Returns:
            None. Files are saved to path.
        """
        df_all = self._obj
        from itertools import product
        fns = list(df_all.metadata.details.keys())
        for fn in fns:
            df = df_all.loc[df_all['filename'] == fn].copy().reset_index(drop = True)
            #Make a multi column object and rearrange table to be in DLC format
            scorer = df_all.metadata.details[fn]['scorer']
            selected_cols = []
            for animal, bp, component in product(df_all.pose.animals, df_all.pose.body_parts, ['x', 'y', 'likelihood']):
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

    @staticmethod
    def load(fn_in : str) -> pd.DataFrame:
        """Load ExperimentDataFrame object from pickle file.
        
        Args:
            fn_in: path to load pickle file from. 
            
        Returns:
            None. Data in this object is populated with contents of file."""
        return load_experiment(fn_in)

    def save_movie(self, label_columns, path_out : str, video_filenames = None) -> None: # pragma: no cover
        """Given columns indicating behavior predictions or whatever else, make a video
        with these predictions overlaid. 

        ExperimentDataFrame metadata must have the keys 'video_file', so that the video associated with each set of pose tracks is known.

        Args:
            label_columns: list or dict of columns whose values to overlay on top of video. If dict, keys are the columns and values are the print-friendly version.
            path_out: the directory to output the videos too
            video_filenames: list or string. The set of videos to use. If not provided, then use all videos as given in the metadata.

        Returns:
            None. Videos are saved to 'path_out'
        """

        if not checkFFMPEG():
            print("Cannot find ffmpeg in path, aborting. Please install.")
            return

        #Text parameters (units in pixels)
        y_offset = global_config['make_movie__y_offset']
        y_inc = global_config['make_movie__y_inc']
        text_color = global_config['make_movie__text_color']
        font_size = global_config['make_movie__font_size']

        if type(video_filenames) is str:
            video_filenames = [video_filenames]
        if not video_filenames:
            video_filenames = self._obj.metadata.videos
        if type(label_columns) is list:
            label_columns = {k:k for k in label_columns}

        for video in video_filenames:
            if 'fps' not in self._obj.metadata.details[video]:
                raise ValueError(f"FPS not in metadata for video {video}, skipping. Provide FPS in dataset creation.")
            rate = 1/self._obj.metadata.details[video]['fps']
            vid_in = self._obj.metadata.details[video]['video_files']
            file_out = os.path.splitext(os.path.basename(vid_in))[0] + '_'.join(label_columns.keys()) + '.mp4'
            vid_out = os.path.join(path_out, file_out)
            label_strings = []
            for idx, (col,print_label) in enumerate(label_columns.items()):
                this_y_offset = y_offset + y_inc*idx
                behavior_pairs = _make_dense_values_into_pairs(self._obj.loc[self._obj.filename == video, col], rate)
                this_label_string = ','.join([f"drawtext=text=\'{print_label}\':x=90:y={this_y_offset}:fontsize={font_size}:fontcolor={text_color}:enable=\'between(t,{pair[0]},{pair[1]})\'" for pair in behavior_pairs])
                label_strings.append(this_label_string)
            label_string = ','.join(label_strings)

            #ffmpeg is the fastest way to add this information to a video
            #Prepare the ffmpeg command
            cmd = f'ffmpeg -y -i {vid_in} -vf "{label_string}" {vid_out}'
            os.system(cmd)            

def _create_from_dict(metadata, label_key, part_renamer, animal_renamer):
    df = pd.DataFrame()
    #req_cols = ['fps']
    #Drop requirement this is provided. Just omit addition of time column, if fps omitted
    req_cols = []
    is_valid, should_rescale = _validate_metadata(metadata, req_cols)
    if is_valid:   
        df.metadata.details = metadata
    else:
        raise ValueError("Metadata not properly formatted. See docstring.")

    if len(metadata) > 0:
        _load_tracks(df, part_renamer, animal_renamer, rescale = should_rescale) 
        _load_labels(df, set_as_label = True)

    if should_rescale:
        _convert_units(df)
    elif 'scale_factor' in df.columns: 
        df.drop(columns = 'scale_factor', inplace = True)

    return df    

def _create_from_list(metadata, label_key, part_renamer, animal_renamer):
    if len(metadata) == 0:
        return pd.DataFrame()
    supported_exts = ['.csv', '.h5', '.nwb', '.hdf5']
    exts = [os.path.splitext(m)[1] for m in metadata]
    print(exts)
    supported = [e in supported_exts for e in exts]
    is_nwb = [e == 'nwb' for e in exts]
    if not all(supported):
        ValueError("Only NWB, dlc (csv) or SLEAP (h5, hdf5) formats are supported.")

    if all(is_nwb):
        df = _load_nwb(metadata, part_renamer, animal_renamer)
    else:
        metadata = {k:{} for k in metadata}
        df = _create_from_dict(metadata, label_key, part_renamer, animal_renamer)

    return df

def create_dataset(metadata : dict = None, 
                     label_key : dict = None, 
                     part_renamer : dict = None,
                     animal_renamer : dict = None) -> pd.DataFrame:
    """Houses pose-tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels.

    Args:
        metadata: Dictionary whose keys are pose tracking files, and value is a dictionary of associated metadata
            for that video. Most easiest to create with 'create_metadata'. 
        label_key: Default None. Dictionary whose keys are positive integers and values are behavior labels. If none, then this is inferred from the behavior annotation files provided.  
        part_renamer: Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.)
        animal_renamer: Default None. Dictionary that can rename animals from tracking files if needed

    Returns:
        DataFrame object. This is a pandas DataFrame with additional metadata and methods.
    """

    if type(metadata) is dict:
        df = _create_from_dict(metadata, label_key, part_renamer, animal_renamer)
    elif type(metadata) is str:
        df = _create_from_list([metadata], label_key, part_renamer, animal_renamer)
    elif type(metadata) is list:
        df = _create_from_list(metadata, label_key, part_renamer, animal_renamer)
    else:
        raise ValueError("Metadata not properly formatted. See docstring.")

    if label_key:
        df.metadata.label_key = label_key

    return df

def _load_nwb(nwb_files, part_renamer, animal_renamer):

    metadata = {}
    dfs = []
    col_names_old = None
    for fn in nwb_files:
        df_fn, body_parts, animals, col_names, scorer, meta = read_NWB_tracks(fn, 
                                                                              part_renamer, 
                                                                              animal_renamer)

        dfs.append(df_fn)
        #Add additional metadata
        metadata[fn] = {}
        metadata[fn]['scorer'] = scorer
        metadata[fn]['video'] = meta[fn]['video_files']
        metadata[fn]['resolution'] = meta[fn]['resolution']
        metadata[fn]['units'] = meta[fn]['unit']

        if col_names_old is not None:
            if col_names != col_names_old:
                raise RuntimeError("NWB files have different animals/body parts tracked. Must all be from same project")
        col_names_old = col_names

    df = pd.concat(dfs, axis = 0)
    #df[dfs.columns] = dfs
    df.reset_index(drop = True, inplace = True)
    df.metadata.details = metadata
    df.pose.body_parts = body_parts
    df.pose.animals = animals
    df.pose.animal_setup = {'mouse_ids': animals, 'bodypart_ids': body_parts, 'colnames': col_names}
    df.pose.raw_track_columns = col_names
    return df

def _load_tracks(df, part_renamer, animal_renamer, rescale = False):
    """Add tracks to DataFrame"""
    dfs = []
    col_names_old = None
    #Go through each video file and load tracks
    for fn in df.metadata.details.keys():
        _, ext = os.path.splitext(fn)
        if ext == '.nwb':
            df_fn, body_parts, animals, col_names, scorer, _ = read_NWB_tracks(fn, 
                                                                            part_renamer, 
                                                                            animal_renamer)
        elif ext == '.csv':
            df_fn, body_parts, animals, col_names, scorer = read_DLC_tracks(fn, 
                                                                            part_renamer, 
                                                                            animal_renamer)
        else:
            #Try DLC first, if this fails, read as SLEAP
            try:
                df_fn, body_parts, animals, col_names, scorer = read_DLC_tracks(fn, 
                                                                                part_renamer, 
                                                                                animal_renamer)
            except:
                df_fn, body_parts, animals, col_names, scorer = read_sleap_tracks(fn, 
                                                                                part_renamer, 
                                                                                animal_renamer)

        n_rows = len(df_fn)
        if 'time' not in df_fn.columns and 'fps' in df.metadata.details[fn]:
            df_fn['time'] = df_fn['frame']/df.metadata.details[fn]['fps']

        if rescale and ('frame_width_units' in df.metadata.details[fn]) and \
                ('frame_width' in df.metadata.details[fn]) and \
                ('resolution' in df.metadata.details[fn]) and \
                df.metadata.details[fn]['frame_width_units'].lower() in UNIT_DICT:
            unit_scale_factor = UNIT_DICT[df.metadata.details[fn]['frame_width_units'].lower()]

            if 'units' in df.metadata.details[fn]:
                unit_scale_factor /= UNIT_DICT[df.metadata.details[fn]['units']]
            else:
                df.metadata.details[fn]['units'] = 'mm'

            df_fn['scale_factor'] = df.metadata.details[fn]['frame_width']/df.metadata.details[fn]['resolution'][1]*unit_scale_factor
        else:
            df_fn['scale_factor'] = 1

        dfs.append(df_fn)
        if 'fps' in df.metadata.details[fn]:
            df.metadata.details[fn]['duration'] = (n_rows)/df.metadata.details[fn]['fps']
        df.metadata.details[fn]['scorer'] = scorer
        if col_names_old is not None:
            if col_names != col_names_old:
                raise RuntimeError("Tracking files have different columns. Must all be from same project")
        col_names_old = col_names

    dfs = pd.concat(dfs, axis = 0)
    df[dfs.columns] = dfs
    df.reset_index(drop = True, inplace = True)
    df.pose.body_parts = body_parts
    df.pose.animals = animals
    df.pose.animal_setup = {'mouse_ids': animals, 'bodypart_ids': body_parts, 'colnames': col_names}
    df.pose.raw_track_columns = col_names
    for animal in df.pose.animals:
        if animal in ['filename', 'frame', 'time', 'label']:
            warnings.warn(f"One of the animal names is protected ({animal}), unexpected behavior may occur.")
    return col_names

def _load_labels(df, col_name = 'label', set_as_label = False):
    #For the moment only BORIS support
    return _load_labels_boris(df, col_name, set_as_label)

def _load_labels_boris(df, col_name = 'label', set_as_label = False):
    """Add behavior label data to DataFrame"""

    if not df.metadata.label_key:
        label_files = []
        for fn in df.metadata.details.keys():
            if 'labels' in df.metadata.details[fn]:
                label_files.append(df.metadata.details[fn]['labels'])
        df.metadata.label_key = create_behavior_labels(label_files)

    label_cols = []
    for vid in df.metadata.details:
        if 'labels' in df.metadata.details[vid] and 'fps' in df.metadata.details[vid]:
            fn_in = df.metadata.details[vid]['labels']
            fps = df.metadata.details[vid]['fps']
            duration = df.metadata.details[vid]['duration']

            ground_truth = read_boris_annotation(fn_in, fps, duration, df.metadata.label_key.values())

            for behavior in ground_truth:
                col_name = 'label_' + behavior
                label_cols.append(col_name)
                if col_name not in df.columns:
                    df[col_name] = 0.
                df.loc[df['filename'] == vid, col_name] = ground_truth[behavior]

    if set_as_label:
        df.ml.label_cols = list(set(label_cols))

def load_experiment(fn_in : str) -> pd.DataFrame:
    """Load DataFrame from file.
    
    Args:
        fn_in: path to file to load
        
    Returns:
        DataFrame object from pickle file
    """
    with open(fn_in, 'rb') as file:
        dataPickle = file.read()
    new_obj = pd.DataFrame({})
    new_obj.__dict__ = dill.loads(dataPickle)
    return new_obj

def get_sample_openfield_data():
    """Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC.
    
    Returns:
        DataFrame with the corresponding tracking and behavior annotation files
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, 'data', 'dlc', 'openfield', '*example.csv'))
    video_files = glob(os.path.join(cur_dir, 'data', 'videos', '*.mp4'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)          # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = create_metadata(tracking_files, 
                            video_files = video_files,
                            fps = fps, 
                            resolution = resolution)

    return create_dataset(metadata)

def _make_dense_values_into_pairs(predictions, rate): # pragma: no cover
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

def add_randomforest_predictions(df : pd.DataFrame):
    """Perform cross validation of a RandomForestClassifier to predict behavior based on 
    activated features. Can be useful to assess model performance, and if you have enough data.
    
    Args:
        df: Dataframe housing features and labels to perform classification. 
        Will perform leave-one-video-out cross validation hence dataframe needs at least two
        videos to run.

    Returns:
        None. Modifies df in place, adding column 'prediction' with model predictions in it.
    """
    cv = LeaveOneGroupOut()
    model = RandomForestClassifier()
    if df.ml.features is None or df.ml.labels is None or df.ml.group is None:
        warnings.warn("features, labels or groups is None, cannot make predictions")
        return

    if len(df.ml.features) == 0 or len(df.ml.labels) == 0 or len(df.ml.group) == 0:
        warnings.warn("features, labels or groups is empyt, cannot make predictions")
        return

    preds = cross_val_predict(model, df.ml.features, df.ml.labels, groups = df.ml.group, cv = cv)
    df['prediction'] = preds

