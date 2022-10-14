import os
import pytest
from glob import glob

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

@pytest.fixture()
def tracking_files():
    print('tracking files:', len(glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv'))))
    return sorted(glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv')))

@pytest.fixture()
def labels():
    print('label files:', len(glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv'))))
    return sorted(glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv')))

@pytest.fixture()
def metadata_params():
    params = {
        'frame_width': 20,
        'frame_width_units': 'in',
        'fps': 30,
        'resolution': (1200, 1600),
    }
    return params

@pytest.fixture()
def sample_nwb_file():
    return os.path.join(TEST_DATA_DIR, 'sample_nwb.nwb')

@pytest.fixture()
def metadata(tracking_files, labels, metadata_params):
    from ethome import create_metadata

    metadata = create_metadata(tracking_files, 
                              labels = labels, 
                              frame_width = metadata_params['frame_width'], 
                              fps = metadata_params['fps'], 
                              frame_width_units = metadata_params['frame_width_units'], 
                              resolution = metadata_params['resolution'])
    return metadata

@pytest.fixture()
def dataset(metadata):
    from ethome import create_dataset
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    edf = create_dataset(metadata, animal_renamer=animal_renamer)
    return edf

@pytest.fixture()
def openfield_sample():
    from ethome.video import get_sample_openfield_data
    return get_sample_openfield_data()

@pytest.fixture
def default_track_cols(dataset):
    return dataset.pose.raw_track_columns

@pytest.fixture()
def videodataset_mars(dataset):
    from ethome import mars_feature_maker
    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    return dataset

@pytest.fixture()
def dens_matrix(dataset):

    from ethome.unsupervised import compute_density
    import numpy as np
    n_pts = 200
    extent = (-50, 50, -50, 50)
    dataset = dataset.iloc[:200,:]
    embedding = np.random.randn(200, 2)
    dataset[['embedding_0', 'embedding_1']] = embedding
    dens_matrix = compute_density(dataset, extent, n_pts = n_pts)
    return dens_matrix