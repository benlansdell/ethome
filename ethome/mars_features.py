import pandas as pd 
from behaveml.dl.feature_engineering import make_features_mars_distr, make_features_social, \
                                            make_features_distances, make_features_velocities, \
                                            make_features_mars_reduced

def compute_mars_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df = make_features_mars_distr(df[raw_col_names], animal_setup) 
    return features_df

def compute_distance_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df = make_features_distances(df[raw_col_names], animal_setup) 
    return features_df

def compute_mars_reduced_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df = make_features_mars_reduced(df[raw_col_names], animal_setup) 
    return features_df

def compute_social_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df = make_features_social(df[raw_col_names], animal_setup) 
    return features_df

def compute_velocity_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df = make_features_velocities(df[raw_col_names], animal_setup) 
    return features_df
