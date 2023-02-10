__version__ = '0.4.0'

#These are brought into the ethome namespace
from ethome.io import read_DLC_tracks, save_DLC_tracks_h5, load_sklearn_model, save_sklearn_model
from ethome.interpolation import interpolate_lowconf_points
from ethome.video import create_dataset, create_metadata, load_experiment, add_randomforest_predictions
from ethome.unsupervised import compute_tsne_embedding
from ethome.features import list_inbuilt_features

#These are the functions imported when doing 'from ethome import *'
__all__ = [create_dataset, create_metadata, read_DLC_tracks, load_experiment, 
           load_sklearn_model, save_sklearn_model, save_DLC_tracks_h5, interpolate_lowconf_points, 
           compute_tsne_embedding, add_randomforest_predictions, list_inbuilt_features]