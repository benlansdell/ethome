import pandas as pd 
import numpy as np
from behaveml.video import VideosetDataFrame

def interpolate_lowconf_points(vdf : VideosetDataFrame,
                               conf_threshold : float = 0.9,
                               in_place : bool = True,
                               rolling_window : bool = True,
                               window_size : int = 3) -> pd.DataFrame:
    """Interpolate raw tracking points if their probabilities are available.

    Args:
        vdf: VideosetDataFrame containing the tracks to interpolate
        conf_threshold: default 0.9. Confidence below which to count as uncertain, and to interpolate its value instead
        in_place: default True. Whether to replace data in place
        rolling_window: default True. Whether to use a rolling window to interpolate
        window_size: default 3. The size of the rolling window to use

    Returns:
        Pandas dataframe with the filtered raw columns. Returns None if opted for in_place modification
    """

    df_filtered = []

    for fn_in in vdf.videos:
        print("processing", fn_in)

        if not in_place:
            df_filter_low_conf = vdf.data.loc[vdf.data.filename == fn_in].copy()
        else:
            df_filter_low_conf = vdf.data
        
        for m in vdf.animals:
            for bp in vdf.body_parts:
                low_conf = df_filter_low_conf.loc[vdf.data.filename == fn_in, '_'.join(['likelihood', m, bp])] < conf_threshold
                df_filter_low_conf.loc[(vdf.data.filename == fn_in) & low_conf,'_'.join([m, 'x', bp])] = np.nan
                df_filter_low_conf.loc[(vdf.data.filename == fn_in) & low_conf,'_'.join([m, 'y', bp])] = np.nan
                
        df_filter_low_conf.loc[(vdf.data.filename == fn_in)] = \
            df_filter_low_conf.loc[(vdf.data.filename == fn_in)].\
            interpolate(axis = 0, method = 'linear', limit_direction = 'both')            

        if rolling_window:
            df_filter_low_conf.loc[(vdf.data.filename == fn_in), vdf.raw_track_columns] = \
                df_filter_low_conf.loc[(vdf.data.filename == fn_in), vdf.raw_track_columns].rolling(window = window_size, min_periods = 1).mean()
            
        df_filter_low_conf = df_filter_low_conf[vdf.raw_track_columns + ['filename', 'frame']]
        df_filtered.append(df_filter_low_conf)

    df_filtered = pd.concat(df_filtered, axis = 0)

    if not in_place:
        return df_filtered
    else:
        return None

#TODO
# Get this working...
# if filter_out_toofast:
#     for m in vdf.animals:
#         for bp in vdf.body_parts:
#             too_quick = (df_filter_low_conf.loc[:,'_'.join([m, 'x', bp])].diff(jump_dur).abs() > speed_threshold) | \
#                         (df_filter_low_conf.loc[:,'_'.join([m, 'y', bp])].diff(jump_dur).abs() > speed_threshold) | \
#                         (df_filter_low_conf.loc[:,'_'.join([m, 'x', bp])].diff(-jump_dur).abs() > speed_threshold) | \
#                         (df_filter_low_conf.loc[:,'_'.join([m, 'y', bp])].diff(-jump_dur).abs() > speed_threshold)

#             df_filter_low_conf.loc[too_quick,'_'.join([m, 'x', bp])] = np.nan
#             df_filter_low_conf.loc[too_quick,'_'.join([m, 'y', bp])] = np.nan

#     #And now filter again
#     df_filter_low_conf = df_filter_low_conf.interpolate(axis = 0, method = 'polynomial', order = 1)
#     df_filter_low_conf = df_filter_low_conf.rolling(window = 3, min_periods = 1).mean()

#Try using the fancyimpute package....
# #df_filter_low_conf = pd.DataFrame(NuclearNormMinimization().fit_transform(df_filter_low_conf.to_numpy()), 
# #                                 index = df_filter_low_conf.index, 
# #                                 columns = df_filter_low_conf.columns)

# df_filter_low_conf = pd.DataFrame(KNN(k=7).fit_transform(df_filter_low_conf.to_numpy()), 
#                                 index = df_filter_low_conf.index, 
#                                 columns = df_filter_low_conf.columns)

# #X_incomplete_normalized = BiScaler().fit_transform(df_filter_low_conf.to_numpy())
# #df_filter_low_conf = pd.DataFrame(SoftImpute().fit_transform(X_incomplete_normalized),
# #                                 index = df_filter_low_conf.index,
# #                                 columns = df_filter_low_conf.columns)
