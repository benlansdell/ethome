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

def test_sample_singlemouse_data():
    from behaveml.video import get_sample_openfield_data
    df = get_sample_openfield_data()
    assert df.animals == ['ind1']
    assert len(df.data.columns) == 15

def test_multiple_boris_behaviors():
    import os 
    from glob import glob 
    from behaveml import VideosetDataFrame, clone_metadata
    import numpy as np

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    label_files = [os.path.join(cur_dir, '..', 'behaveml', 'data', 'boris', 'multiple_behaviors', 'e3v813a-20210610T120637-121213_reencode_multiple_behaviors.csv'),
                   os.path.join(cur_dir, '..', 'behaveml', 'data', 'boris', 'e3v813a-20210610T121558-122141_reencode.csv')]
    tracking_files = [os.path.join(cur_dir, '..', 'behaveml', 'data', 'dlc', 'e3v813a-20210610T120637-121213DLC_dlcrnetms5_pilot_studySep24shuffle1_100000_el_filtered.csv'),
                      os.path.join(cur_dir, '..', 'behaveml', 'data', 'dlc', 'e3v813a-20210610T121558-122141DLC_dlcrnetms5_pilot_studySep24shuffle1_100000_el_filtered.csv')]

    fps = 30                         # (int) frames per second
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = clone_metadata(tracking_files, 
                              label_files = label_files,
                              fps = fps)

    vdf = VideosetDataFrame(metadata)

    assert 'interact' in vdf.label_key.values()
    assert 'mount' in vdf.label_key.values()
    assert all(np.array(list(vdf.label_key.keys())) > 0)

def test_sample_singlemouse_data_missing():

    import os 
    from glob import glob 
    from behaveml import VideosetDataFrame, clone_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, '..', 'behaveml', 'data', 'dlc', 'openfield', '*missingdata.csv'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)        # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = clone_metadata(tracking_files, 
                            fps = fps, 
                            resolution = resolution)

    vdf = VideosetDataFrame(metadata)

    assert vdf.animals == ['ind1']
    assert len(vdf.data.columns) == 15

def test_sample_threemouse_data():

    import os 
    from glob import glob 
    from behaveml import VideosetDataFrame, clone_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, '..', 'behaveml', 'data', 'dlc', 'openfield', '*three.csv'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)          # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = clone_metadata(tracking_files, 
                            fps = fps, 
                            resolution = resolution)

    vdf = VideosetDataFrame(metadata)

    assert vdf.animals == ['ind1', 'ind2', 'ind3']
    assert len(vdf.data.columns) == 39