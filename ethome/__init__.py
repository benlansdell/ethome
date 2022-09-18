__version__ = '0.2.9'

#These are brought into the ethome namespace
from ethome.io import read_DLC_tracks, save_DLC_tracks_h5, load_sklearn_model, save_sklearn_model
from ethome.interpolation import interpolate_lowconf_points
from ethome.video import createExperiment, clone_metadata, load_videodataset

from ethome.features.features import cnn_probability_feature_maker, mars_feature_maker, social_feature_maker, \
                              distance_feature_maker, speed_feature_maker, marsreduced_feature_maker, \
                              com_interanimal_feature_maker, com_interanimal_speed_feature_maker, \
                              com_feature_maker, com_velocity_feature_maker

from ethome.unsupervised import compute_tsne_embedding

#These are the functions imported when doing 'from ethome import *'
__all__ = [createExperiment, clone_metadata, read_DLC_tracks, load_videodataset, 
           load_sklearn_model, save_sklearn_model, save_DLC_tracks_h5, 
           mars_feature_maker, cnn_probability_feature_maker, 
           distance_feature_maker, speed_feature_maker, social_feature_maker,
           marsreduced_feature_maker, interpolate_lowconf_points, compute_tsne_embedding,
           com_interanimal_feature_maker, com_interanimal_speed_feature_maker, \
           com_feature_maker, com_velocity_feature_maker, \
           distance_feature_maker]