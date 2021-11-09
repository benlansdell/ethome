""" Functions to take pose tracks and compute a set of features from them """
import pandas as pd

from behaveml.dl.dl_features import compute_dl_probability_features

def create_mars_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict):
    return pd.DataFrame(df) 