import os
import pytest
from glob import glob

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

@pytest.fixture()
def tracking_files():
    print('tracking files:', len(glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv'))))
    return glob(os.path.join(TEST_DATA_DIR, 'dlc/*.csv'))

@pytest.fixture()
def label_files():
    print('label files:', len(glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv'))))
    return glob(os.path.join(TEST_DATA_DIR, 'boris/*.csv'))

@pytest.fixture()
def metadata_params():
    params = {
        'frame_length': None,
        'units': None,
        'fps': 30,
        'resolution': (1200, 1600),
    }
    return params