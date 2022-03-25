#####################
## Setup test code ##
#####################

import pytest

def test_sample_data():
    from behaveml.io import get_sample_data
    from behaveml import VideosetDataFrame
    dataset = get_sample_data()
    assert type(dataset) is VideosetDataFrame

def test_sample_paths():
    from behaveml.io import get_sample_data_paths
    tracking_files, boris_files = get_sample_data_paths()
    assert type(tracking_files) is list
    assert type(boris_files) is list
    assert len(tracking_files) == len(boris_files)
