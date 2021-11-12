""" Loading and saving tracking and behavior annotation files """
import pandas as pd
import pickle
import numpy as np
from itertools import product

XY_IDS = ['x', 'y']

def _uniquifier(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def read_DLC_tracks(fn_in : str):
    """Read in tracks from DLC.

    Args:
        fn_in: (str) csv file that has DLC tracks

    Returns:
        Pandas DataFrame with (n_animals*2*n_body_parts) columns, plus with filename and frame
    """
    df = pd.read_csv(fn_in, header = [0,1,2,3], index_col = 0)
    df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])

    cols = list(df.columns)
    animals = _uniquifier([i[1] for i in cols])
    body_parts = _uniquifier([i[2] for i in cols])

    n_body_parts = len(body_parts)
    n_animals = len(animals)

    dlc_tracks = np.array(df)
    n_rows = dlc_tracks.shape[0]
    selected_cols = [[3*i, 3*i+1] for i in range(n_body_parts*n_animals)]
    #This line flattens the array here... may be clearer using numpy reshape?
    selected_cols = [j for i in selected_cols for j in i]
    dlc_tracks = dlc_tracks[:,selected_cols]

    #Put in shape:
    # (frame, animal, x/y coord, body part)
    dlc_tracks = dlc_tracks.reshape((n_rows, n_animals, n_body_parts, 2))
    dlc_tracks = dlc_tracks.transpose([0, 1, 3, 2])

    colnames = ['_'.join(a) for a in product(animals, XY_IDS, body_parts)]

    dlc_tracks = dlc_tracks.reshape((n_rows, -1))
    final_df = pd.DataFrame(dlc_tracks, columns = colnames)
    final_df['filename'] = fn_in
    final_df['frame'] = final_df.index.copy()

    return final_df, body_parts, animals, colnames

def save_DLC_tracks_h5(df : pd.DataFrame, fn_out : str):
    df.to_hdf(fn_out, "df_with_missing", format = 'table', mode="w")

def load_data(fn : str):
    """Load an object from a pickle file"""
    try:
        with open(fn, 'rb') as handle:
            a = pickle.load(handle)
    except FileNotFoundError:
        print("Cannot find", fn)
        return None
    return a 