""" Loading and saving tracking and behavior annotation files """
import pandas as pd
import pickle
import numpy as np
from itertools import product
from joblib import dump, load
import os
from glob import glob
import warnings

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

def save_sklearn_model(model, fn_out):
    """Save sklearn model to file
    
    Args:
        model: sklearn model to save
        fn_out: filename to save to
    """
    dump(model, fn_out) 

def load_sklearn_model(fn_in):
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
                    read_likelihoods : bool = True) -> tuple:
    """Read in tracks from DLC.

    Args:
        fn_in: csv file that has DLC tracks
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

    df = pd.read_csv(fn_in, header = [0,1,2,3], index_col = 0)

    if 'individuals' in df.columns.names:
        df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])
        multi_animal = True
    else:
        df = pd.read_csv(fn_in, header = [0,1,2], index_col = 0)
        df.columns = df.columns.set_names(['scorer', 'bodyparts', 'coords'])
        multi_animal = False

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

    return final_df, body_parts, animals, colnames, scorer

def save_DLC_tracks_h5(df : pd.DataFrame, fn_out : str) -> None:
    """Save DLC tracks in h5 format.
    
    Args:
        df: Pandas dataframe to save
        fn_out: Where to save the dataframe
    """
    df.to_hdf(fn_out, "df_with_missing", format = 'table', mode="w")

def load_data(fn : str):
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
    from ethome import createExperiment, clone_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = sorted(glob(cur_dir + '/data/dlc/*.csv'))
    boris_files = sorted(glob(cur_dir + '/data/boris/*.csv'))
    frame_width = None               # (float) length of entire horizontal shot
    frame_width_units = None         # (str) units frame_width is given in
    fps = 30                         # (int) frames per second
    resolution = (1200, 1600)        # (tuple) HxW in pixels
    metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

    dataset = createExperiment(metadata)
    path_out = os.path.join(cur_dir, 'data', fn_out)
    to_save = {'dataset': dataset, 'metadata': metadata}
    with open(path_out, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    path_in = os.path.join(cur_dir, 'data', 'sample_dataframe.pkl')
    with open(path_in, 'rb') as handle:
        b = pickle.load(handle)
    return b['dataset']

def read_boris_annotation(fn_in : str, fps : int, duration : float, behav_labels : dict = None) -> tuple:
    """Read behavior annotation from BORIS exported csv file. 

    This will import behavior types specified (or all types, if behavior_list is None) and assign a numerical label to each. Overlapping annotations (those occurring simulataneously) are not supported. Any time the video is annotated as being in multiple states, the last state will be the one labeled.
    
    Args:
        fn_in: The filename with BORIS behavior annotations to load
        fps: The frames per second of the video
        duration: The duration of the video in seconds
        behav_labels: If provided, only import behaviors with these names. Default = None = import everything. 
    
    Returns:
        A numpy array which indicates, for all frames, which behavior is occuring. 0 = no behavior, 1 and above are the labels of the behaviors.
        A dictionary with keys the numerical labels and values the names of the behaviors. 
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
    ground_truth = np.zeros(n_bins)

    if behav_labels is None:
        behaviors = boris_labels['Behavior'].unique()
        behav_labels = {i+1:k for i,k in enumerate(behaviors)}

    for behav_idx, behavior in behav_labels.items():
        labels = boris_labels[boris_labels['Behavior'] == behavior]
        starts = labels.loc[labels['Status'] == 'START', 'Time']
        ends = labels.loc[labels['Status'] == 'STOP', 'Time']
        if len(ends) > len(starts)+1:
            raise ValueError(f"Too many {behavior} behaviors started and not stopped.")
        elif len(ends) == len(starts) + 1:
            ends.append(duration)
        for start, end in zip(starts, ends):
            ground_truth[int(start*fps):int(end*fps)] = behav_idx

    return ground_truth, behav_labels

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