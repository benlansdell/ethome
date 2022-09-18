import pandas as pd 
import numpy as np

def interpolate_lowconf_points(edf : pd.DataFrame,
                               conf_threshold : float = 0.9,
                               in_place : bool = True,
                               rolling_window : bool = True,
                               window_size : int = 3) -> pd.DataFrame:
    """Interpolate raw tracking points if their probabilities are available.

    Args:
        edf: pandas DataFrame containing the tracks to interpolate
        conf_threshold: default 0.9. Confidence below which to count as uncertain, and to interpolate its value instead
        in_place: default True. Whether to replace data in place
        rolling_window: default True. Whether to use a rolling window to interpolate
        window_size: default 3. The size of the rolling window to use

    Returns:
        Pandas dataframe with the filtered raw columns. Returns None if opted for in_place modification
    """

    df_filtered = []

    for fn_in in edf.metadata.videos:
        print("processing", fn_in)

        if not in_place:
            df_filter_low_conf = edf.loc[edf.filename == fn_in].copy()
        else:
            df_filter_low_conf = edf
        
        for m in edf.pose.animals:
            for bp in edf.pose.body_parts:
                low_conf = \
                    df_filter_low_conf.loc[edf.filename == fn_in, '_'.join(['likelihood', m, bp])] < conf_threshold
                df_filter_low_conf.loc[(edf.filename == fn_in) & low_conf,'_'.join([m, 'x', bp])] = np.nan
                df_filter_low_conf.loc[(edf.filename == fn_in) & low_conf,'_'.join([m, 'y', bp])] = np.nan
                
        df_filter_low_conf.loc[(edf.filename == fn_in)] = \
            df_filter_low_conf.loc[(edf.filename == fn_in)].\
            interpolate(axis = 0, method = 'linear', limit_direction = 'both')            

        if rolling_window:
            df_filter_low_conf.loc[(edf.filename == fn_in), edf.pose.raw_track_columns] = \
                df_filter_low_conf.loc[(edf.filename == fn_in), edf.pose.raw_track_columns].rolling(window = window_size, min_periods = 1).mean()
            
        df_filter_low_conf = df_filter_low_conf[edf.pose.raw_track_columns + ['filename', 'frame']]
        df_filtered.append(df_filter_low_conf)

    df_filtered = pd.concat(df_filtered, axis = 0)

    if not in_place:
        return df_filtered
    else:
        return None