__version__ = '0.3.0'

#These are brought into the ethome namespace
from ethome.io import read_DLC_tracks, save_DLC_tracks_h5, load_sklearn_model, save_sklearn_model
from ethome.interpolation import interpolate_lowconf_points
from ethome.video import createExperiment, clone_metadata, load_videodataset
from ethome.unsupervised import compute_tsne_embedding

#These are the functions imported when doing 'from ethome import *'
__all__ = [createExperiment, clone_metadata, read_DLC_tracks, load_videodataset, 
           load_sklearn_model, save_sklearn_model, save_DLC_tracks_h5, interpolate_lowconf_points, 
           compute_tsne_embedding]