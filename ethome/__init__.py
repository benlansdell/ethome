__version__ = '0.2.9'

#These are brought into the behaveml namespace
from behaveml.io import read_DLC_tracks, save_DLC_tracks_h5, load_sklearn_model, save_sklearn_model
from behaveml.interpolation import interpolate_lowconf_points
from behaveml.video import VideosetDataFrame, clone_metadata, load_videodataset

from behaveml.features import cnn_probability_feature_maker, mars_feature_maker, social_feature_maker, \
                              distance_feature_maker, speed_feature_maker, marsreduced_feature_maker, \
                              com_interanimal_feature_maker, com_interanimal_speed_feature_maker, \
                              com_feature_maker, com_velocity_feature_maker

from behaveml.unsupervised import compute_tsne_embedding

#These are the functions imported when doing 'from behaveml import *'
__all__ = [VideosetDataFrame, clone_metadata, read_DLC_tracks, load_videodataset, 
           load_sklearn_model, save_sklearn_model, save_DLC_tracks_h5, 
           mars_feature_maker, cnn_probability_feature_maker, 
           distance_feature_maker, speed_feature_maker, social_feature_maker,
           marsreduced_feature_maker, interpolate_lowconf_points, compute_tsne_embedding,
           com_interanimal_feature_maker, com_interanimal_speed_feature_maker, \
           com_feature_maker, com_velocity_feature_maker, \
           distance_feature_maker]