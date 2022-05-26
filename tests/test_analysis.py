#####################
## Setup test code ##
#####################

import pytest

from behaveml import VideosetDataFrame, clone_metadata, interpolate_lowconf_points, video
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
    try: df = VideosetDataFrame(metadata)
    except: assert False, "Failed to make VideosetDataFrame object"

    try: df = VideosetDataFrame({})
    except: assert False, "Failed to make empty VideosetDataFrame object"

    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])

    try: df = VideosetDataFrame(metadata_no_labels)
    except: assert False, "Failed to make VideosetDataFrame object without labels"

    #Also check that improper formatted metadata raises the right exception

    #Check that 

def test_VideoDataFrame_object(videodataset):
    assert videodataset.feature_cols is None

def test_df_renaming(metadata, default_track_cols):
    none_renamer = {}
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = VideosetDataFrame(metadata, part_renamer = none_renamer, animal_renamer=animal_renamer)
    df.feature_cols = df.raw_track_columns
    assert df.feature_cols == default_track_cols

    new_parts = ['nose', 'left_ear', 'right_ear', 'neck', 'lefthip', 'righthip', 'tail']
    part_renamer = {'leftear': 'left_ear', 'rightear': 'right_ear'}
    df = VideosetDataFrame(metadata, part_renamer = part_renamer)
    assert set(new_parts) == set(df.body_parts)

    new_animals = ['resident', 'intruder']
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = VideosetDataFrame(metadata, animal_renamer = animal_renamer)
    assert set(new_animals) == set(df.animals)

def test_dl_features(videodataset):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from behaveml import cnn_probability_feature_maker
    videodataset.add_features(cnn_probability_feature_maker, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)
    assert set(videodataset.feature_cols) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_dl_features_with_missing(videodataset):
    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from behaveml import cnn_probability_feature_maker
    videodataset.data.iloc[:10,0] = np.nan
    videodataset.add_features(cnn_probability_feature_maker, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)
    assert set(videodataset.feature_cols) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_add_likelihood(videodataset):
    new_cols = videodataset.activate_features_by_name('likelihood')
    assert len(new_cols) == 14

def test_remove_likelihood(videodataset):
    new_cols = videodataset.activate_features_by_name('likelihood')
    old_cols = videodataset.remove_features_by_name('likelihood')
    assert new_cols == old_cols

def test_mars_features(videodataset):
    from behaveml import mars_feature_maker
    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 726

def test_mars_features_with_missing(videodataset):
    import numpy as np
    from behaveml import mars_feature_maker
    #Give the df missing data:
    videodataset.data.iloc[:10,0] = np.nan

    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 726

def test_mars_then_interpolate(videodataset):
    from behaveml import mars_feature_maker
    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    interpolate_lowconf_points(videodataset)
    assert len(videodataset.feature_cols) == 726

def test_singleanimal_interpolate(openfield_sample):
    interpolate_lowconf_points(openfield_sample)
    assert len(openfield_sample.data.columns) == 15

def test_duplicate_mars_features(videodataset):
    from behaveml import mars_feature_maker
    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    videodataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 726

def test_distance_features(videodataset):
    from behaveml import distance_feature_maker
    videodataset.add_features(distance_feature_maker, 
                     featureset_name = 'distances', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 91

def test_velocity_features(videodataset):
    from behaveml import velocity_feature_maker
    videodataset.add_features(velocity_feature_maker, 
                     featureset_name = 'speeds', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 91

def test_social_features(videodataset):
    from behaveml import social_feature_maker
    videodataset.add_features(social_feature_maker, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 98

def test_marsreduced_features(videodataset):
    from behaveml import marsreduced_feature_maker
    videodataset.add_features(marsreduced_feature_maker, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(videodataset.feature_cols) == 285

def test_interpolate(videodataset):
    interpolate_lowconf_points(videodataset)

    assert pd.notnull(videodataset.data).all(axis = None)

def test_rescaling_metadataparams(tracking_files, metadata_params):
    """ Test creation of VideoDataFrame object with more varied metadata key combinations"""

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_width = clone_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = VideosetDataFrame(metadata_no_width)
    assert 'units' not in list(df.metadata.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_widthunits = clone_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                resolution = metadata_params['resolution'])
    df = VideosetDataFrame(metadata_no_widthunits)
    assert 'units' not in list(df.metadata.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we should
    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = VideosetDataFrame(metadata_no_labels)
    assert list(df.metadata.values())[0]['units'] == 'mm'

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    # Because we're just missing one value in one rec
    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    del list(metadata_no_labels.values())[1]['frame_width']
    df = VideosetDataFrame(metadata_no_labels)
    assert 'units' not in list(df.metadata.values())[0]
