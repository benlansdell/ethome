#####################
## Setup test code ##
#####################

import pytest

from ethome import createExperiment, clone_metadata, interpolate_lowconf_points, video
import pandas as pd

#Metadata is a dictionary
def test_clone_metadata(tracking_files, label_files, metadata_params):
    """ Test creation of metadata object """

    metadata = clone_metadata(tracking_files, 
                                label_files = label_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])

    assert len(metadata) == 5
    assert metadata[list(metadata.keys())[0]]['fps'] == metadata_params['fps']



def test_VideoDataFrame(tracking_files, label_files, metadata, metadata_params):
    """ Test creation of VideoDataFrame object """
    # metadata = clone_metadata(tracking_files, 
    #                             label_files = label_files, 
    #                             frame_width = metadata_params['frame_width'], 
    #                             fps = metadata_params['fps'], 
    #                             frame_width_units = metadata_params['frame_width_units'], 
    #                             resolution = metadata_params['resolution'])

    #Eventually, check we can make it without error
    try: df = createExperiment(metadata)
    except: assert False, "Failed to make createExperiment object"

    try: df = createExperiment({})
    except: assert False, "Failed to make empty createExperiment object"

    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])

    try: df = createExperiment(metadata_no_labels)
    except: assert False, "Failed to make createExperiment object without labels"

    #Also check that improper formatted metadata raises the right exception

    #Check that 

def test_VideoDataFrame_object(dataset):
    assert dataset.features.active is None

def test_df_renaming(metadata, default_track_cols):
    none_renamer = {}
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = createExperiment(metadata, part_renamer = none_renamer, animal_renamer=animal_renamer)
    df.features.active = df.pose.raw_track_columns
    assert df.features.active == default_track_cols

    new_parts = ['nose', 'left_ear', 'right_ear', 'neck', 'lefthip', 'righthip', 'tail']
    part_renamer = {'leftear': 'left_ear', 'rightear': 'right_ear'}
    df = createExperiment(metadata, part_renamer = part_renamer)
    assert set(new_parts) == set(df.pose.body_parts)

    new_animals = ['resident', 'intruder']
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = createExperiment(metadata, animal_renamer = animal_renamer)
    assert set(new_animals) == set(df.pose.animals)

def test_dl_features(dataset):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from ethome import cnn_probability_feature_maker
    dataset.features.add(cnn_probability_feature_maker, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)
    assert set(dataset.features.active) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_dl_features_with_missing(dataset):
    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from ethome import cnn_probability_feature_maker
    dataset.iloc[:10,0] = np.nan
    dataset.features.add(cnn_probability_feature_maker, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)
    assert set(dataset.features.active) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_add_likelihood(dataset):
    new_cols = dataset.features.activate('likelihood')
    assert len(new_cols) == 14

def test_remove_likelihood(dataset):
    new_cols = dataset.features.activate('likelihood')
    old_cols = dataset.features.deactivate('likelihood')
    assert new_cols == old_cols

def test_mars_features(dataset):
    from ethome import mars_feature_maker
    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_mars_features_with_missing(dataset):
    import numpy as np
    from ethome import mars_feature_maker
    # 
    #Give the df missing data:
    dataset.iloc[:10,0] = np.nan

    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_mars_then_interpolate(dataset):
    from ethome import mars_feature_maker
    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    interpolate_lowconf_points(dataset)
    assert len(dataset.features.active) == 726

def test_singleanimal_interpolate(openfield_sample):
    interpolate_lowconf_points(openfield_sample)
    assert len(openfield_sample.columns) == 15

def test_duplicate_mars_features(dataset):
    from ethome import mars_feature_maker
    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    dataset.features.add(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_distance_features(dataset):
    from ethome import distance_feature_maker
    dataset.features.add(distance_feature_maker, 
                     featureset_name = 'distances', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_speed_features(dataset):
    from ethome import speed_feature_maker
    dataset.features.add(speed_feature_maker, 
                     featureset_name = 'speeds', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_social_features(dataset):
    from ethome import social_feature_maker
    dataset.features.add(social_feature_maker, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 98

def test_marsreduced_features(dataset):
    from ethome import marsreduced_feature_maker
    dataset.features.add(marsreduced_feature_maker, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 285

def test_interpolate(dataset):
    interpolate_lowconf_points(dataset)

    assert pd.notnull(dataset).all(axis = None)

def test_rescaling_metadataparams(tracking_files, metadata_params):
    """ Test creation of VideoDataFrame object with more varied metadata key combinations"""

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_width = clone_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = createExperiment(metadata_no_width)
    assert 'units' not in list(df.metadata.details.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_widthunits = clone_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                resolution = metadata_params['resolution'])
    df = createExperiment(metadata_no_widthunits)
    assert 'units' not in list(df.metadata.details.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we should
    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = createExperiment(metadata_no_labels)
    assert list(df.metadata.details.values())[0]['units'] == 'mm'

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    # Because we're just missing one value in one rec
    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    del list(metadata_no_labels.values())[1]['frame_width']
    df = createExperiment(metadata_no_labels)
    assert 'units' not in list(df.metadata.details.values())[0]

def test_save_to_dlc_csv(dataset, tmp_path_factory):
    import os
    from glob import glob
    fn = tmp_path_factory.mktemp('dlc_csv')
    dataset.io.to_dlc_csv(fn)
    created_files = glob(os.path.join(fn, '*.csv'))
    assert len(created_files) > 0

###########################
## Test generic features ##
###########################

def test_centerofmass_interanimal_features(dataset):
    from ethome import com_interanimal_feature_maker
    dataset.features.add(com_interanimal_feature_maker, 
                     featureset_name = 'com_interanimal', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 1

def test_centerofmass_interanimal_speed_features(dataset):
    from ethome import com_interanimal_speed_feature_maker
    dataset.features.add(com_interanimal_speed_feature_maker, 
                     featureset_name = 'com_interanimal_speed', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 1

def test_centerofmass_features(dataset):
    from ethome import com_feature_maker
    dataset.features.add(com_feature_maker, 
                     featureset_name = 'com', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 4

def test_centerofmass_vel_features(dataset):
    from ethome import com_velocity_feature_maker
    dataset.features.add(com_velocity_feature_maker, 
                     featureset_name = 'com_vel', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 4

def test_speed_features(dataset):
    from ethome import speed_feature_maker
    dataset.features.add(speed_feature_maker, 
                     featureset_name = 'speed', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_dist_features(dataset):
    from ethome import distance_feature_maker
    dataset.features.add(distance_feature_maker, 
                     featureset_name = 'dist', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91