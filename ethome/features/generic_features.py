""" Functions to take pose tracks and compute a set of features from them """

import pandas as pd
import numpy as np

def _diff_within_group(df, sort_key, diff_col, **kwargs):
    return df.groupby(sort_key)[diff_col].transform(lambda x: x.diff(**kwargs)) 

def compute_centerofmass_interanimal_distances(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    """Distances between all animals' centroids"""
    animal_setup = df.pose.animal_setup
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()

    for animal_id in mouse_ids:
        fxs = ['_'.join([animal_id, 'x', bp]) for bp in bodypart_ids]
        fys = ['_'.join([animal_id, 'y', bp]) for bp in bodypart_ids]
        fx_new = '_'.join([animal_id, 'COM_x'])
        fy_new = '_'.join([animal_id, 'COM_y'])
        features_df[fx_new] = features_df[fxs].sum(axis = 1) / len(bodypart_ids)
        features_df[fy_new] = features_df[fys].sum(axis = 1) / len(bodypart_ids)

    orig_cols = features_df.columns

    for i, animal_i in enumerate(mouse_ids):
        fx_i = '_'.join([animal_i, 'COM_x'])
        fy_i = '_'.join([animal_i, 'COM_y'])
        for j, animal_j in enumerate(mouse_ids[:i]):
            fx_j = '_'.join([animal_j, 'COM_x'])
            fy_j = '_'.join([animal_j, 'COM_y'])
            f_new = '_'.join([animal_i, animal_j, 'COM_distance'])
            features_df[f_new] = np.sqrt((features_df[fx_i] - features_df[fx_j])**2 \
                                         + (features_df[fy_i] - features_df[fy_j])**2)

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_centerofmass_interanimal_speed(df : pd.DataFrame, raw_col_names : list, n_shifts = 5, **kwargs) -> pd.DataFrame:
    """Speeds between all animals' centroids"""
    animal_setup = df.pose.animal_setup
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()

    dt = features_df['time'].diff(periods = n_shifts)

    for animal_id in mouse_ids:
        fxs = ['_'.join([animal_id, 'x', bp]) for bp in bodypart_ids]
        fys = ['_'.join([animal_id, 'y', bp]) for bp in bodypart_ids]
        fx_new = '_'.join([animal_id, 'COM_x'])
        fy_new = '_'.join([animal_id, 'COM_y'])
        features_df[fx_new] = features_df[fxs].sum(axis = 1) / len(bodypart_ids)
        features_df[fy_new] = features_df[fys].sum(axis = 1) / len(bodypart_ids)

    orig_cols = features_df.columns

    for i, animal_i in enumerate(mouse_ids):
        fx_i = '_'.join([animal_i, 'COM_x'])
        fy_i = '_'.join([animal_i, 'COM_y'])
        for _, animal_j in enumerate(mouse_ids[:i]):
            fx_j = '_'.join([animal_j, 'COM_x'])
            fy_j = '_'.join([animal_j, 'COM_y'])
            f_new = '_'.join([animal_i, animal_j, 'COM_speed'])

            vx_i = _diff_within_group(features_df, 'filename', fx_i, periods = n_shifts)/dt
            vy_i = _diff_within_group(features_df, 'filename', fy_i, periods = n_shifts)/dt
            vx_j = _diff_within_group(features_df, 'filename', fx_j, periods = n_shifts)/dt
            vy_j = _diff_within_group(features_df, 'filename', fy_j, periods = n_shifts)/dt

            features_df[f_new] = np.sqrt((vx_i - vx_j)**2 + (vy_i - vy_j)**2)

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_centerofmass(df : pd.DataFrame, raw_col_names : list, bodyparts : list = [], **kwargs) -> pd.DataFrame:
    """Centroid of all animals"""
    animal_setup = df.pose.animal_setup
    if len(bodyparts) == 0:
        bodypart_ids = animal_setup['bodypart_ids']
    else:
        bodypart_ids = [v for v in bodyparts if v in animal_setup['bodypart_ids']]
    if len(bodypart_ids) == 0:
        raise ValueError('No listed bodyparts found in animal_setup')

    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    for animal_id in mouse_ids:
        fxs = ['_'.join([animal_id, 'x', bp]) for bp in bodypart_ids]
        fys = ['_'.join([animal_id, 'y', bp]) for bp in bodypart_ids]
        fx_new = '_'.join([animal_id, 'COM_x'])
        fy_new = '_'.join([animal_id, 'COM_y'])
        features_df[fx_new] = features_df[fxs].sum(axis = 1) / len(bodypart_ids)
        features_df[fy_new] = features_df[fys].sum(axis = 1) / len(bodypart_ids)

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_centerofmass_velocity(df : pd.DataFrame, raw_col_names : list, n_shifts = 5, bodyparts : list = [], **kwargs) -> pd.DataFrame:
    """Velocity of all animals' centroids"""
    animal_setup = df.pose.animal_setup

    if len(bodyparts) == 0:
        bodypart_ids = animal_setup['bodypart_ids']
    else:
        bodypart_ids = [v for v in bodyparts if v in animal_setup['bodypart_ids']]
    if len(bodypart_ids) == 0:
        raise ValueError('No listed bodyparts found in animal_setup')

    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    dt = features_df['time'].diff(periods = n_shifts)

    for animal_id in mouse_ids:
        fxs = ['_'.join([animal_id, 'x', bp]) for bp in bodypart_ids]
        fys = ['_'.join([animal_id, 'y', bp]) for bp in bodypart_ids]
        fx_new = '_'.join([animal_id, 'COM_vel_x'])
        fy_new = '_'.join([animal_id, 'COM_vel_y'])
        features_df[fx_new] = features_df[fxs].sum(axis = 1) / len(bodypart_ids)
        features_df[fy_new] = features_df[fys].sum(axis = 1) / len(bodypart_ids)
        features_df[fx_new] = _diff_within_group(features_df, 'filename', fx_new, periods = n_shifts)/dt
        features_df[fy_new] = _diff_within_group(features_df, 'filename', fy_new, periods = n_shifts)/dt

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_part_velocity(df : pd.DataFrame, raw_col_names : list, n_shifts = 5, bodyparts : list = [], **kwargs) -> pd.DataFrame:
    """Velocity of all animals' bodyparts"""
    animal_setup = df.pose.animal_setup

    if len(bodyparts) == 0:
        bodypart_ids = animal_setup['bodypart_ids']
    else:
        bodypart_ids = [v for v in bodyparts if v in animal_setup['bodypart_ids']]
    if len(bodypart_ids) == 0:
        raise ValueError('No listed bodyparts found in animal_setup')

    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    dt = features_df['time'].diff(periods = n_shifts)

    for animal_id in mouse_ids:
        for bp in bodypart_ids:
            fx = '_'.join([animal_id, 'x', bp])
            fy = '_'.join([animal_id, 'y', bp])
            fx_new = '_'.join([animal_id, bp, 'vel_x'])
            fy_new = '_'.join([animal_id, bp, 'vel_y'])
            features_df[fx_new] = _diff_within_group(features_df, 'filename', fx, periods = n_shifts)/dt
            features_df[fy_new] = _diff_within_group(features_df, 'filename', fy, periods = n_shifts)/dt

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_part_speed(df : pd.DataFrame, raw_col_names : list, n_shifts = 5, bodyparts : list = [], **kwargs) -> pd.DataFrame:
    """Speed of all animals' bodyparts"""
    animal_setup = df.pose.animal_setup

    if len(bodyparts) == 0:
        bodypart_ids = animal_setup['bodypart_ids']
    else:
        bodypart_ids = [v for v in bodyparts if v in animal_setup['bodypart_ids']]
    if len(bodypart_ids) == 0:
        raise ValueError('No listed bodyparts found in animal_setup')

    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    dt = features_df['time'].diff(periods = n_shifts)

    for animal_id in mouse_ids:
        for bp in bodypart_ids:
            fx = '_'.join([animal_id, 'x', bp])
            fy = '_'.join([animal_id, 'y', bp])
            x_new = _diff_within_group(features_df, 'filename', fx, periods = n_shifts)/dt
            y_new = _diff_within_group(features_df, 'filename', fy, periods = n_shifts)/dt
            f_new = '_'.join([animal_id, bp, 'speed'])
            features_df[f_new] = np.sqrt(x_new**2 + y_new**2)

    features_df = features_df.drop(columns = orig_cols)
    return features_df

def compute_speed_features(df : pd.DataFrame, raw_col_names : list, n_shifts = 5, **kwargs) -> pd.DataFrame:
    """Speeds between all body parts pairs (within and between animals)"""
    animal_setup = df.pose.animal_setup
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    dt = features_df['time'].diff(periods = n_shifts)

    ##Make the distance features
    for i, bp1 in enumerate(bodypart_ids):
        for j, bp2 in enumerate(bodypart_ids):
            if i < j:
                for mouse_id in mouse_ids:
                    #We can compute the intra-mouse difference
                    f1x = '_'.join([mouse_id, 'x', bp1])
                    f2x = '_'.join([mouse_id, 'x', bp2])
                    f1y = '_'.join([mouse_id, 'y', bp1])
                    f2y = '_'.join([mouse_id, 'y', bp2])
                    f_new = '_'.join([mouse_id, 'speed', bp1, bp2])

                    vx1 = _diff_within_group(features_df, 'filename', f1x, periods = n_shifts)/dt
                    vy1 = _diff_within_group(features_df, 'filename', f1y, periods = n_shifts)/dt
                    vx2 = _diff_within_group(features_df, 'filename', f2x, periods = n_shifts)/dt
                    vy2 = _diff_within_group(features_df, 'filename', f2y, periods = n_shifts)/dt
                    features_df[f_new] = np.sqrt((vx1 - vx2)**2 + (vy1 - vy2)**2)

            #Inter-mouse difference
            for animal_i in range(len(mouse_ids)):
                for animal_j in range(animal_i):
                    f1x = '_'.join([mouse_ids[animal_i], 'x', bp1])
                    f2x = '_'.join([mouse_ids[animal_j], 'x', bp2])
                    f1y = '_'.join([mouse_ids[animal_i], 'y', bp1])
                    f2y = '_'.join([mouse_ids[animal_j], 'y', bp2])
                    f_new = '_'.join([f'M{animal_i}_M{animal_j}', 'speed', bp1, bp2])

                    vx1 = _diff_within_group(features_df, 'filename', f1x, periods = n_shifts)/dt
                    vy1 = _diff_within_group(features_df, 'filename', f1y, periods = n_shifts)/dt
                    vx2 = _diff_within_group(features_df, 'filename', f2x, periods = n_shifts)/dt
                    vy2 = _diff_within_group(features_df, 'filename', f2y, periods = n_shifts)/dt
                    features_df[f_new] = np.sqrt((vx1 - vx2)**2 + (vy1 - vy2)**2)

    #Remove base features
    features_df = features_df.drop(columns = orig_cols)

    return features_df

def compute_distance_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    """Distances between all body parts pairs (within and between animals)"""
    animal_setup = df.pose.animal_setup

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']

    features_df = df.copy()
    orig_cols = df.columns

    ##Make the distance features
    for i, bp1 in enumerate(bodypart_ids):
        for j, bp2 in enumerate(bodypart_ids):
            if i < j:
                for mouse_id in mouse_ids:
                    #We can compute the intra-mouse difference
                    f1x = '_'.join([mouse_id, 'x', bp1])
                    f2x = '_'.join([mouse_id, 'x', bp2])
                    f1y = '_'.join([mouse_id, 'y', bp1])
                    f2y = '_'.join([mouse_id, 'y', bp2])
                    f_new = '_'.join([mouse_id, 'distance', bp1, bp2])
                    features_df[f_new] = \
                        np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                (features_df[f1y] - features_df[f2y])**2)

            #Inter-mouse difference
            for animal_i in range(len(mouse_ids)):
                for animal_j in range(animal_i):
                    f1x = '_'.join([mouse_ids[animal_i], 'x', bp1])
                    f2x = '_'.join([mouse_ids[animal_j], 'x', bp2])
                    f1y = '_'.join([mouse_ids[animal_i], 'y', bp1])
                    f2y = '_'.join([mouse_ids[animal_j], 'y', bp2])
                    f_new = '_'.join([f'M{animal_i}_M{animal_j}', 'distance', bp1, bp2])
                    features_df[f_new] = \
                                np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                        (features_df[f1y] - features_df[f2y])**2)

    #Remove base features
    features_df = features_df.drop(columns = orig_cols)

    return features_df