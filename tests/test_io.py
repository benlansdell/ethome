#####################
## Setup test code ##
#####################

import pandas as pd

# def test_nwb_dandiset231_import():
#     from ethome import create_dataset
#     path = '/home/blansdel/projects/ethome/ethome/data/sample_nwb_dandiset_231.nwb'
#     dataset = create_dataset(path)
#     assert type(dataset) is pd.DataFrame

def test_dataset_creation_list(tracking_files):

    from ethome.io import get_sample_nwb_paths
    from ethome import create_dataset
    import os 
    from glob import glob 

    #Create dataset from openfield nwb and openfield DLC
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    openfield_path = glob(os.path.join(cur_dir, '..', 'ethome', 'data', 'dlc', 'openfield', '*missingdata.csv'))[0]

    nwb_path = get_sample_nwb_paths()
    dataset = create_dataset([nwb_path, openfield_path], animal_renamer = {'PoseEstimation':'ind1'})
    assert type(dataset) is pd.DataFrame

    dataset = create_dataset(tracking_files)
    assert type(dataset) is pd.DataFrame

def test_nwb_import():
    from ethome.io import get_sample_nwb_paths
    from ethome import create_dataset
    path = get_sample_nwb_paths()
    dataset = create_dataset(path)
    assert type(dataset) is pd.DataFrame

def test_sleap_import(sleap_file):
    from ethome.io import read_sleap_tracks
    df = read_sleap_tracks(sleap_file)
    assert type(df[0]) is pd.DataFrame
    assert df[1] == ['abdomen', 'eyeL', 'eyeR', 'forelegL4', 'forelegR4', 'head', 'hindlegL4', 'hindlegR4', 'midlegL4', 'midlegR4', 'thorax', 'wingL', 'wingR']
    assert df[2] == ['Animal0', 'Animal1']

def test_sample_data():
    from ethome.io import get_sample_data
    dataset = get_sample_data()
    assert type(dataset) is pd.DataFrame

def test_sample_paths():
    from ethome.io import get_sample_data_paths
    tracking_files, boris_files = get_sample_data_paths()
    assert type(tracking_files) is list
    assert type(boris_files) is list
    assert len(tracking_files) == len(boris_files)

def test_sample_singlemouse_data():
    from ethome.video import get_sample_openfield_data
    df = get_sample_openfield_data()
    assert df.pose.animals == ['ind1']
    assert len(df.columns) == 15

def test_sample_simultaneous_behavior_data(tracking_files, simultaneous_labels):
    import numpy as np
    from ethome import create_dataset, create_metadata
    metadata = create_metadata(tracking_files[:1], 
                               labels = simultaneous_labels,
                               fps = 30)

    df = create_dataset(metadata)
    assert 2 in np.unique(df.ml.labels.sum(axis = 1))

def test_multiple_boris_behaviors():
    import os 
    from glob import glob 
    from ethome import create_dataset, create_metadata
    import numpy as np

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    label_files = [os.path.join(cur_dir, '..', 'ethome', 'data', 'boris', 'multiple_behaviors', 'e3v813a-20210610T120637-121213_reencode_multiple_behaviors.csv'),
                   os.path.join(cur_dir, '..', 'ethome', 'data', 'boris', 'e3v813a-20210610T121558-122141_reencode.csv')]
    tracking_files = [os.path.join(cur_dir, '..', 'ethome', 'data', 'dlc', 'e3v813a-20210610T120637-121213DLC_dlcrnetms5_pilot_studySep24shuffle1_100000_el_filtered.csv'),
                      os.path.join(cur_dir, '..', 'ethome', 'data', 'dlc', 'e3v813a-20210610T121558-122141DLC_dlcrnetms5_pilot_studySep24shuffle1_100000_el_filtered.csv')]

    fps = 30                         # (int) frames per second
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = create_metadata(tracking_files, 
                              labels = label_files,
                              fps = fps)

    edf = create_dataset(metadata)

    assert 'interact' in edf.metadata.label_key.values()
    assert 'mount' in edf.metadata.label_key.values()
    assert all(np.array(list(edf.metadata.label_key.keys())) > 0)

def test_sample_singlemouse_data_missing():

    import os 
    from glob import glob 
    from ethome import create_dataset, create_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, '..', 'ethome', 'data', 'dlc', 'openfield', '*missingdata.csv'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)          # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = create_metadata(tracking_files, 
                            fps = fps, 
                            resolution = resolution)

    edf = create_dataset(metadata)

    assert edf.pose.animals == ['ind1']
    assert len(edf.columns) == 15

def test_sample_threemouse_data():

    import os 
    from glob import glob 
    from ethome import create_dataset, create_metadata

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_files = glob(os.path.join(cur_dir, '..', 'ethome', 'data', 'dlc', 'openfield', '*three.csv'))
    fps = 30                         # (int) frames per second
    resolution = (480, 640)          # (tuple) HxW in pixels
    #Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
    metadata = create_metadata(tracking_files, 
                            fps = fps, 
                            resolution = resolution)

    edf = create_dataset(metadata)

    assert edf.pose.animals == ['ind1', 'ind2', 'ind3']
    assert len(edf.columns) == 39