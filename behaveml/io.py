""" Loading and saving tracking and behavior annotation files """
import pandas as pd
import pickle
import numpy as np
from itertools import product
from joblib import dump, load
import os
from glob import glob

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
    dump(model, fn_out) 

def load_sklearn_model(fn_in):
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
    df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])

    scorer = df.columns.get_level_values(0)[0]

    cols = list(df.columns)
    animals = uniquifier([i[1] for i in cols])
    body_parts = uniquifier([i[2] for i in cols])

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

def rename_df_cols(df : pd.DataFrame, renamer : dict) -> pd.DataFrame:
    """Rename dataframe columns 
    
    Args:
        df: Pandas dataframe whose columns to rename
        renamer: dictionary whose key:value pairs define the substitutions to make

    Returns:
        The dataframe with renamed columns.
    """
    return df.rename(columns = renamer)

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

def _make_sample_dataframe(fn_out = 'sample_dataframe.pkl'):
    from behaveml import VideosetDataFrame, clone_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = sorted(glob(cur_dir + '/data/dlc/*.csv'))
    boris_files = sorted(glob(cur_dir + '/data/boris/*.csv'))
    frame_length = None              # (float) length of entire horizontal shot
    units = None                     # (str) units frame_length is given in
    fps = 30                         # (int) frames per second
    resolution = (1200, 1600)        # (tuple) HxW in pixels
    metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          frame_length = frame_length, 
                          fps = fps, 
                          units = units, 
                          resolution = resolution)

    dataset = VideosetDataFrame(metadata)
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
        (VideosetDataFrame) Data frame with the corresponding tracking and behavior annotation files
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_in = os.path.join(cur_dir, 'data', 'sample_dataframe.pkl')
    with open(path_in, 'rb') as handle:
        b = pickle.load(handle)
    return b['dataset']

def read_boris_annotation(fn_in : str, fps : int, duration : float) -> np.ndarray:
    """Read behavior annotation from BORIS exported csv file
    
    Args:
        fn_in: The filename with BORIS behavior annotations to load
        fps: Frames per second of video
        duration: Duration of video
    
    Returns:
        A numpy array which indicates, for all frames, if behavior is occuring (1) or not (0)
    """
    n_bins = int(duration*fps)
    boris_labels = pd.read_csv(fn_in, skiprows = 15)
    boris_labels['index'] = (boris_labels.index//2)
    boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
    boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
    boris_labels = [list(i) for i in boris_labels]
    ground_truth = np.zeros(n_bins)
    for start, end in boris_labels:
        ground_truth[int(start*fps):int(end*fps)] = 1
    return ground_truth