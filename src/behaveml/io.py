""" Loading and saving tracking and behavior annotation files """
import pandas as pd

def read_DLC_tracks(fn_in):
    """Read in tracks from DLC"""
    df = pd.read_csv(fn_in, header = [0,1,2,3], index_col = 0)
    df = df.set_index(('scorer', 'individuals', 'bodyparts', 'coords'))
    df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])
    return df

def save_DLC_tracks_h5(df, fn_out):
    df.to_hdf(fn_out, "df_with_missing", format = 'table', mode="w")

def load_data(fn):
    """Load an object from a pickle file"""
    try:
        with open(fn, 'rb') as handle:
            a = pickle.load(handle)
    except FileNotFoundError:
        print("Cannot find", fn)
        return None
    return a 