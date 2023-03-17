""" Loading and saving tracking and behavior annotation files """
import pandas as pd
import pickle
import numpy as np
from itertools import product
from joblib import dump, load
import os
import h5py
from glob import glob
import warnings
from pynwb import NWBHDF5IO
from collections import defaultdict

XY_IDS = ['x', 'y']
XYLIKELIHOOD_IDS = ['x', 'y', 'likelihood']

def uniquifier(seq):
    """Return a sequence (e.g. list) with unique elements only, but maintaining original list order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def _list_replace(ls, renamer):
    """Replace elements in a list according to provided dictionary"""
    for i, word in enumerate(ls):
        if word in renamer.keys():
            ls[i] = renamer[word]
    return ls

def save_sklearn_model(model, fn_out): # pragma: no cover
    """Save sklearn model to file
    
    Args:
        model: sklearn model to save
        fn_out: filename to save to
    """
    dump(model, fn_out) 

def load_sklearn_model(fn_in): # pragma: no cover
    """Load sklearn model from file
    
    Args:
        fn_in: filename to load from
    
    Returns:
        the loaded sklearn model
    """
    model = load(fn_in)
    return model 

def read_DLC_tracks(fn_in : str, 
                    part_renamer : dict = None, 
                    animal_renamer : dict = None,
                    read_likelihoods : bool = True,
                    labels : pd.DataFrame = None) -> tuple:
    """Read in tracks from DLC.

    Args:
        fn_in: csv or h5 file that has DLC tracks
        part_renamer: dictionary to rename body parts, if needed 
        animal_renamer: dictionary to rename animals, if needed
        read_likelihoods: default True. Whether to attach DLC likelihoods to table

    Returns:
        Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame, 
            List of body parts,
            List of animals,
            Columns names for DLC tracks (excluding likelihoods, if read in),
            Scorer
    """

    ext = os.path.splitext(fn_in)[1]
    if ext == 'h5' or ext == 'hdf5' or ext == 'hdf':
        df = pd.read_hdf(fn_in)
        ext = 'h5'
    else:
        df = pd.read_csv(fn_in, header = [0,1,2,3], index_col = 0)
        ext = 'csv'
    return _read_DLC_tracks(df, fn_in, part_renamer, animal_renamer, read_likelihoods, labels, ext)

def read_sleap_tracks(fn_in : str, 
                    part_renamer : dict = None, 
                    animal_renamer : dict = None,
                    read_likelihoods : bool = True,
                    labels : pd.DataFrame = None) -> tuple:
    """Read in tracks from SLEAP.

    Args:
        fn_in: csv or h5 file that has sleap tracks
        part_renamer: dictionary to rename body parts, if needed 
        animal_renamer: dictionary to rename animals, if needed
        read_likelihoods: default True. Whether to attach DLC likelihoods to table

    Returns:
        Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame, 
            List of body parts,
            List of animals,
            Columns names for DLC tracks (excluding likelihoods, if read in),
            Scorer
    """

    df = _load_sleap_data(fn_in, multi_animal=False)
    return _read_sleap_tracks(df, fn_in, part_renamer, animal_renamer, read_likelihoods, labels)

## This is from here
## https://github.com/YttriLab/A-SOID/blob/d03302b3d35e22b0348b9f199998f293d8d741cd/asoid/utils/import_data.py
## Under the following license:
# The Clear BSD License
# Copyright (c) 2022, YttriLab, SchwarzLab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted for academic and research use only (subject to the limitations in the disclaimer below) provided that the following conditions are met:

#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.

#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.

#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def _load_sleap_data(path, multi_animal=False):
    """loads sleap data h5 format from file path and returns it as pd.DataFrame
    As sleap body parts (nodes) are not ordered in a particular way, we sort them alphabetically.
    As sleap tracks do not export a score/likelihood but cut off automatically (nan), we are simulating a likelihood
     value for compatability with feature extraction functions"""
    # TODO: discriminate with multiple animal from single instance model output
    with h5py.File(path, "r") as f:
        # occupancy_matrix = f['track_occupancy'][:]
        # track_names = f['track_names'][:]
        tracks_matrix = f["tracks"][:].T
        animals = []
        temp_tracks = None
        for an in np.arange(tracks_matrix.shape[3]):
            animals.append('Animal{}'.format(an))
            if temp_tracks is None:
                temp_tracks = tracks_matrix[:, :, :, an]
            else:
                temp_tracks = np.concatenate([temp_tracks, tracks_matrix[:, :, :, an]], axis=1)

        bodyparts = f["node_names"][:].astype("U13")
    # preallocate array including dummy likelihood for similarity between dlc and sleap import
    # can be changed later if sleap files incorporate scores at any time in the future
    likelihood_matrix = np.ones((temp_tracks.shape[0], temp_tracks.shape[1], 1))
    sleap_matrix = np.concatenate((temp_tracks, likelihood_matrix), axis=2)
    # # reshape tracks into 2d format (x,y, likelihood)*len(bps)
    sleap_array = np.reshape(sleap_matrix, (sleap_matrix.shape[0], sleap_matrix.shape[1] * sleap_matrix.shape[2]))
    # create beautiful multiindex columns to easily sort by bodypart

    column_names = pd.MultiIndex.from_product([animals, bodyparts, ('x', 'y', 'likelihood')],
                                              names=['individuals', 'bodypart', 'coords'])

    # create dataframe from both
    df = pd.DataFrame(sleap_array, columns=column_names)
    # #change any nan values (lost tracking) to 0
    # TODO: confirm that this is the way to go
    for i, ani in enumerate(animals):
        for bp in bodyparts:
            # find any nan either in x or y coordinates. Usually they are paired, but you never know!
            any_nan = np.any(pd.isna(df[[(ani, bp, 'x'), (ani, bp, 'x')]]).values, axis=1)
            # get the indexes of the boolean mask
            nan_indexes = np.argwhere(any_nan).flatten()
            # set all nan index to 0
            df[(ani, bp, 'likelihood')][nan_indexes] = 0

    # SLEAP comes with NaN values, so we need to get rid of them!
    # first linear interpolation
    clean_df = df.interpolate(method="linear", limit_direction='forward')
    # backward fill with first valid entry to take care of NaNs in the beginning
    clean_df.fillna(method="bfill", inplace=True)
    # sort bodyparts by alphabet
    clean_df = clean_df.reindex(columns=sorted(bodyparts), level=1)
    return clean_df

def _read_sleap_tracks(df : pd.DataFrame, 
                       fn_in : str, 
                       part_renamer : dict = None, 
                       animal_renamer : dict = None,
                       read_likelihoods : bool = True,
                       labels : pd.DataFrame = None) -> tuple:

    if 'time' in df.columns:
        times = df['time'].reset_index(drop = True)
        df.drop(columns = 'time', inplace = True)
    else:
        times = None

    scorer = 'sleap'

    cols = list(df.columns)

    animals = uniquifier([i[0] for i in cols])
    body_parts = uniquifier([i[1] for i in cols])

    n_body_parts = len(body_parts)
    n_animals = len(animals)

    dlc_tracks = np.array(df)
    n_rows = dlc_tracks.shape[0]
    selected_cols = [[3*i, 3*i+1] for i in range(n_body_parts*n_animals)]
    #This line flattens the array here... may be clearer using numpy reshape?
    selected_cols = [j for i in selected_cols for j in i]

    prob_cols = [3*i+2 for i in range(n_body_parts*n_animals)]
    dlc_probs = dlc_tracks[:,prob_cols]
    dlc_tracks = dlc_tracks[:,selected_cols]

    #Put in shape:
    # (frame, animal, x/y coord, body part)
    dlc_tracks = dlc_tracks.reshape((n_rows, n_animals, n_body_parts, 2))
    dlc_tracks = dlc_tracks.transpose([0, 1, 3, 2])

    dlc_probs = dlc_probs.reshape((n_rows, n_animals, n_body_parts, 1))
    dlc_probs = dlc_probs.transpose([0, 3, 1, 2])

    #If we're going to rename items in the list, do it here
    if part_renamer:
        body_parts = _list_replace(body_parts, part_renamer)

    if animal_renamer:
        animals = _list_replace(animals, animal_renamer)

    colnames = ['_'.join(a) for a in product(animals, XY_IDS, body_parts)]
    prob_colnames = ['_'.join(a) for a in product(['likelihood'], animals, body_parts)]

    dlc_probs = dlc_probs.reshape((n_rows, -1))
    final_probs = pd.DataFrame(dlc_probs, columns = prob_colnames)

    dlc_tracks = dlc_tracks.reshape((n_rows, -1))
    final_df = pd.DataFrame(dlc_tracks, columns = colnames)

    if read_likelihoods:
        final_df = pd.concat([final_df, final_probs], axis = 1)

    final_df['filename'] = fn_in
    final_df['frame'] = final_df.index.copy()
    if times is not None:
        final_df['time'] = times

    if labels is not None:
        labels.index = final_df.index
        final_df = pd.concat((final_df, labels), axis = 1)
        final_df = final_df.rename(columns = {k:k[0] for k in labels.columns})
        final_df.ml.label_cols = [p[0] for p in labels.columns]

    return final_df, body_parts, animals, colnames, scorer

def _read_DLC_tracks(df : pd.DataFrame, 
                     fn_in : str,
                     part_renamer : dict = None, 
                     animal_renamer : dict = None,
                     read_likelihoods : bool = True,
                     labels : pd.DataFrame = None, 
                     ext : str = 'csv') -> tuple:

    if 'individuals' in df.columns.names:
        df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])
        multi_animal = True
    else:
        if ext == 'csv':
            df = pd.read_csv(fn_in, header = [0,1,2], index_col = 0)
        else:
            df = pd.read_hdf(fn_in)
        df.columns = df.columns.set_names(['scorer', 'bodyparts', 'coords'])
        multi_animal = False

    if 'time' in df.columns:
        times = df['time'].reset_index(drop = True)
        df.drop(columns = 'time', inplace = True)
    else:
        times = None

    scorer = df.columns.get_level_values(0)[0]

    cols = list(df.columns)

    if multi_animal:
        animals = uniquifier([i[1] for i in cols])
        body_parts = uniquifier([i[2] for i in cols])
    else:
        animals = ['ind1']
        body_parts = uniquifier([i[1] for i in cols])

    n_body_parts = len(body_parts)
    n_animals = len(animals)

    dlc_tracks = np.array(df)
    n_rows = dlc_tracks.shape[0]
    selected_cols = [[3*i, 3*i+1] for i in range(n_body_parts*n_animals)]
    #This line flattens the array here... may be clearer using numpy reshape?
    selected_cols = [j for i in selected_cols for j in i]

    prob_cols = [3*i+2 for i in range(n_body_parts*n_animals)]
    dlc_probs = dlc_tracks[:,prob_cols]
    dlc_tracks = dlc_tracks[:,selected_cols]

    #Put in shape:
    # (frame, animal, x/y coord, body part)
    dlc_tracks = dlc_tracks.reshape((n_rows, n_animals, n_body_parts, 2))
    dlc_tracks = dlc_tracks.transpose([0, 1, 3, 2])

    dlc_probs = dlc_probs.reshape((n_rows, n_animals, n_body_parts, 1))
    dlc_probs = dlc_probs.transpose([0, 3, 1, 2])

    #If we're going to rename items in the list, do it here
    if part_renamer:
        body_parts = _list_replace(body_parts, part_renamer)

    if animal_renamer:
        animals = _list_replace(animals, animal_renamer)

    colnames = ['_'.join(a) for a in product(animals, XY_IDS, body_parts)]
    prob_colnames = ['_'.join(a) for a in product(['likelihood'], animals, body_parts)]

    dlc_probs = dlc_probs.reshape((n_rows, -1))
    final_probs = pd.DataFrame(dlc_probs, columns = prob_colnames)

    dlc_tracks = dlc_tracks.reshape((n_rows, -1))
    final_df = pd.DataFrame(dlc_tracks, columns = colnames)

    if read_likelihoods:
        final_df = pd.concat([final_df, final_probs], axis = 1)

    final_df['filename'] = fn_in
    final_df['frame'] = final_df.index.copy()
    if times is not None:
        final_df['time'] = times

    if labels is not None:
        labels.index = final_df.index
        final_df = pd.concat((final_df, labels), axis = 1)
        final_df = final_df.rename(columns = {k:k[0] for k in labels.columns})
        final_df.ml.label_cols = [p[0] for p in labels.columns]

    return final_df, body_parts, animals, colnames, scorer

## This function is from DLC2NWB package: https://github.com/DeepLabCut/DLC2NWB/blob/10331daa1bfadb9c19d2e4957aa8752d74d5759b/dlc2nwb/utils.py#L307 
#It's available under the following license:
# MIT License

# Copyright (c) 2021- DeepLabCut

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def _convert_nwb_to_h5_all(nwbfile):
    """
    Convert a NWB data file back to DeepLabCut's h5 data format.

    Args:
        nwbfile (str): Path to the newly created NWB data file

    Returns:
        df (pandas.array): Pandas multi-column array containing predictions in DLC format.
    """
    meta = {}

    with NWBHDF5IO(nwbfile, mode="r", load_namespaces=True) as io:
        read_nwbfile = io.read()

        object_keys = read_nwbfile.processing["behavior"].data_interfaces.keys()
        objects = [read_nwbfile.processing["behavior"].data_interfaces[k] for \
                    k in object_keys if 'PoseEstimation' in str(type(read_nwbfile.processing["behavior"].data_interfaces[k]))]

        object_dfs = []
        for read_pe in objects:
            scorer = read_pe.scorer or "scorer"
            dfs = []
            animal = read_pe.name
            animal_key = nwbfile
            videos = read_pe.original_videos[:]
            if len(videos) == 1:
                videos = videos[0]
            meta[animal_key] = {'resolution': np.squeeze(read_pe.dimensions[:]),
                                'video_files': videos,
                                'scorer': scorer}
            for node in read_pe.nodes:
                pes = read_pe.pose_estimation_series[node]
                meta[animal_key]['unit'] = pes.unit
                _, kpt = node.split("_")
                data = pes.data*pes.conversion
                array = np.c_[data, pes.confidence]
                cols = pd.MultiIndex.from_product(
                    [[scorer], [animal], [kpt], ["x", "y", "likelihood"]],
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )
                dfs.append(
                    pd.DataFrame(array, np.around(np.asarray(pes.timestamps), 3), cols)
                )
            object_dfs.append(pd.concat(dfs, axis=1))
        df = pd.concat(object_dfs, axis=1)
        df['time'] = df.index.copy()

        ##Read in behavior labels
        object_keys = read_nwbfile.processing["behavior"].data_interfaces.keys()
        objects = [read_nwbfile.processing["behavior"].data_interfaces[k] for \
                    k in object_keys if 'epoch.TimeIntervals' in str(type(read_nwbfile.processing["behavior"].data_interfaces[k]))]
        new_df = df.copy()

        cols = []
        for read_pe in objects:
            name = read_pe.name
            col = 'label_' + name
            cols.append(col)
            new_df[col] = 0
            start_times = read_pe.start_time[:]
            stop_times = read_pe.stop_time[:]
            pairs = sorted(zip(start_times, stop_times))
            for idx, time in enumerate(new_df.index):
                if len(pairs) == 0: break 
                if time >= pairs[0][0] and time <= pairs[0][1]:
                    new_df.loc[time,col] = 1
                elif time > pairs[0][1]:
                    pairs.pop(0)
            
        new_df = new_df[cols]

    return df, new_df, meta

def read_NWB_tracks(fn_in : str, 
                    part_renamer : dict = None, 
                    animal_renamer : dict = None,
                    read_likelihoods : bool = True) -> tuple:
    """Read in tracks from NWB PoseEstimiationSeries format (something saved using the DLC2NWB package).

    Args:
        fn_in: nwb file that has the tracking information
        part_renamer: dictionary to rename body parts, if needed 
        animal_renamer: dictionary to rename animals, if needed
        read_likelihoods: default True. Whether to attach DLC likelihoods to table

    Returns:
        Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame, 
            List of body parts,
            List of animals,
            Columns names for pose tracks (excluding likelihoods, if read in),
            Scorer
    """
    df, df_labels, metadata = _convert_nwb_to_h5_all(fn_in)
    return _read_DLC_tracks(df, fn_in, part_renamer, animal_renamer, read_likelihoods, df_labels) + (metadata,)

def save_DLC_tracks_h5(df : pd.DataFrame, fn_out : str) -> None: # pragma: no cover
    """Save DLC tracks in h5 format.
    
    Args:
        df: Pandas dataframe to save
        fn_out: Where to save the dataframe
    """
    df.to_hdf(fn_out, "df_with_missing", format = 'table', mode="w")

def load_data(fn : str): # pragma: no cover
    """Load an object from a pickle file
    
    Args:
        fn: The filename

    Returns:
        The pickled object.
    """
    with open(fn, 'rb') as handle:
        object = pickle.load(handle)
    return object 

#Only used to making a test dataframe for testing and dev purposes
def _make_sample_dataframe(fn_out = 'sample_dataframe.pkl'): # pragma: no cover
    from ethome import create_dataset, create_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = sorted(glob(cur_dir + '/data/dlc/*.csv'))
    boris_files = sorted(glob(cur_dir + '/data/boris/*.csv'))
    frame_width = None               # (float) length of entire horizontal shot
    frame_width_units = None         # (str) units frame_width is given in
    fps = 30                         # (int) frames per second
    resolution = (1200, 1600)        # (tuple) HxW in pixels
    metadata = create_metadata(tracking_files, 
                          labels = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

    dataset = create_dataset(metadata)
    path_out = os.path.join(cur_dir, 'data', fn_out)
    to_save = {'dataset': dataset, 'metadata': metadata}
    with open(path_out, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_sample_nwb_paths():
    """Get path to a sample NWB file with tracking data for testing and dev purposes.
    
    Returns:
        Path to a sample NWB file.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, 'data/sample_nwb_.nwb')

def get_sample_data_paths():
    """Get path to sample data files provided with package. 
    
    Returns:
        (tuple) list of DLC tracking file, list of boris annotation files
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = sorted(glob(cur_dir + '/data/dlc/*.csv'))
    boris_files = sorted(glob(cur_dir + '/data/boris/*.csv'))
    return tracking_files, boris_files

def get_sample_data():
    """Load a sample dataset of 5 mice social interaction videos. Each video is approx. 5 minutes in duration
    
    Returns:
        (ExperimentDataFrame) Data frame with the corresponding tracking and behavior annotation files
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_in = os.path.join(cur_dir, 'data', 'sample_data.pkl')
    with open(path_in, 'rb') as handle:
        b = pickle.load(handle)
        #b = pd.read_pickle(handle)
    return b

def read_boris_annotation(fn_in : str, fps : int, duration : float, behav_labels : list = None) -> tuple:
    """Read behavior annotation from BORIS exported csv file. 

    This will import behavior types specified (or all types, if behavior_list is None) and assign a numerical label to each. Overlapping annotations (those occurring simulataneously) are not supported. Any time the video is annotated as being in multiple states, the last state will be the one labeled.
    
    Args:
        fn_in: The filename with BORIS behavior annotations to load
        fps: The frames per second of the video
        duration: The duration of the video in seconds
        behav_labels: If provided, only import behaviors with these names. Default = None = import everything. 
    
    Returns:
        A dictionary of numpy arrays which gives, for all frames, which behavior is occuring. 0 = no behavior, 1 = behavior.
    """

    boris_labels = pd.read_csv(fn_in, skiprows = 15)
    if len(boris_labels) == 0:
        print("No data found in BORIS file,", fn_in)
        return np.array([]), {}
    fps_boris = int(boris_labels['FPS'][0])
    duration_boris = boris_labels['Total length'][0]
    if fps_boris != fps:
        warnings.warn(f"Warning: BORIS FPS is {fps_boris} but video is {fps} frames per second. Are the DLC and BORIS files from the same video?")
    if not np.isclose(duration_boris, duration, rtol = 0.01, atol = 0.01):
        warnings.warn(f"Warning: BORIS duration is {duration_boris} but video is {duration} seconds. Are the DLC and BORIS files from the same video?")

    n_bins = int(duration*fps)
    ground_truth = defaultdict(lambda: np.zeros(n_bins))

    if behav_labels is None:
        behav_labels = boris_labels['Behavior'].unique()
    
    for behavior in behav_labels:
        labels = boris_labels[boris_labels['Behavior'] == behavior]
        starts = labels.loc[labels['Status'] == 'START', 'Time']
        ends = labels.loc[labels['Status'] == 'STOP', 'Time']
        if len(ends) > len(starts)+1:
            raise ValueError(f"Too many {behavior} behaviors started and not stopped.")
        elif len(ends) == len(starts) + 1:
            ends.append(duration)
        for start, end in zip(starts, ends):
            ground_truth[behavior][int(start*fps):int(end*fps)] = 1

    return ground_truth

def create_behavior_labels(boris_files):
    """Create behavior labels from BORIS exported csv files.
    
    Args:
        boris_files: List of BORIS exported csv files
        
    Returns:
        A dictionary with keys the numerical labels and values the names of the behaviors.
    """
    behaviors = set()
    for fn in boris_files:
        boris_labels = pd.read_csv(fn, skiprows = 15)
        behaviors = behaviors | set(boris_labels['Behavior'].unique())
    behavior_labels = {i+1:k for i,k in enumerate(behaviors)}
    return behavior_labels