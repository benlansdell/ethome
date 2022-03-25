import os
import pytest
from glob import glob

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

@pytest.fixture()
def tracking_files():
    print('tracking files:', len(glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv'))))
    return sorted(glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv')))

@pytest.fixture()
def label_files():
    print('label files:', len(glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv'))))
    return sorted(glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv')))

@pytest.fixture()
def metadata_params():
    params = {
        'frame_width': None,
        'frame_width_units': None,
        'fps': 30,
        'resolution': (1200, 1600),
    }
    return params

@pytest.fixture()
def metadata(tracking_files, label_files, metadata_params):
    from behaveml import clone_metadata

    metadata = clone_metadata(tracking_files, 
                              label_files = label_files, 
                              frame_width = metadata_params['frame_width'], 
                              fps = metadata_params['fps'], 
                              frame_width_units = metadata_params['frame_width_units'], 
                              resolution = metadata_params['resolution'])
    return metadata

@pytest.fixture()
def videodataset(metadata):
    from behaveml import VideosetDataFrame
    vdf = VideosetDataFrame(metadata)
    return vdf

@pytest.fixture
def default_track_cols(videodataset):
    return videodataset.raw_track_columns

@pytest.fixture()
def videodataset_mars(videodataset):
    from behaveml import mars_feature_maker
    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    return videodataset

@pytest.fixture()
def dens_matrix(videodataset):

    from behaveml.unsupervised import compute_density
    import numpy as np
    n_pts = 200
    extent = (-50, 50, -50, 50)
    videodataset.data = videodataset.data.iloc[:200,:]
    embedding = np.random.randn(200, 2)
    videodataset.data[['embedding_0', 'embedding_1']] = embedding
    dens_matrix = compute_density(videodataset, extent, n_pts = n_pts)
    return dens_matrix