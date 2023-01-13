#####################
## Setup test code ##
#####################

import pytest

from ethome import create_dataset, create_metadata, interpolate_lowconf_points, video
import pandas as pd
import warnings

#Metadata is a dictionary
def test_create_metadata(tracking_files, labels, metadata_params):
    """ Test creation of metadata object """

    metadata = create_metadata(tracking_files, 
                                labels = labels, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])

    assert len(metadata) == 5
    assert metadata[list(metadata.keys())[0]]['fps'] == metadata_params['fps']



def test_VideoDataFrame(tracking_files, labels, metadata, metadata_params):
    """ Test creation of DataFrame object """
    # metadata = create_metadata(tracking_files, 
    #                             label_files = label_files, 
    #                             frame_width = metadata_params['frame_width'], 
    #                             fps = metadata_params['fps'], 
    #                             frame_width_units = metadata_params['frame_width_units'], 
    #                             resolution = metadata_params['resolution'])

    #Eventually, check we can make it without error
    try: df = create_dataset(metadata)
    except: assert False, "Failed to make create_dataset object"

    try: df = create_dataset({})
    except: assert False, "Failed to make empty create_dataset object"

    metadata_no_labels = create_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])

    try: df = create_dataset(metadata_no_labels)
    except: assert False, "Failed to make create_dataset object without labels"

def test_videodataframe_scaling_raises_error(tracking_files, labels, metadata_params):

    with pytest.warns(Warning):
        metadata = create_metadata(tracking_files, 
                                    labels = labels, 
                                    frame_width = 'a', 
                                    fps = metadata_params['fps'], 
                                    frame_width_units = metadata_params['frame_width_units'], 
                                    resolution = metadata_params['resolution'])
        df = create_dataset(metadata)

    with pytest.warns(Warning):
        metadata = create_metadata(tracking_files, 
                                    labels = labels, 
                                    frame_width = metadata_params['frame_width'], 
                                    fps = metadata_params['fps'], 
                                    frame_width_units = 12, 
                                    resolution = metadata_params['resolution'])
        df = create_dataset(metadata)

    with pytest.warns(Warning):
        metadata = create_metadata(tracking_files, 
                                    labels = labels, 
                                    frame_width = metadata_params['frame_width'], 
                                    fps = metadata_params['fps'], 
                                    frame_width_units = 'asdf', 
                                    resolution = metadata_params['resolution'])
        df = create_dataset(metadata)

    with pytest.warns(Warning):
        metadata = create_metadata(tracking_files, 
                                    labels = labels, 
                                    frame_width = metadata_params['frame_width'], 
                                    fps = metadata_params['fps'], 
                                    frame_width_units = metadata_params['frame_width_units'], 
                                    resolution = (1))
        df = create_dataset(metadata)

    with pytest.warns(Warning):
        metadata = create_metadata(tracking_files, 
                                    labels = labels, 
                                    frame_width = metadata_params['frame_width'], 
                                    fps = metadata_params['fps'], 
                                    frame_width_units = metadata_params['frame_width_units'], 
                                    resolution = metadata_params['resolution'],
                                    units = 'cm')
        metadata[list(metadata.keys())[0]]['units'] = 'mm'
        df = create_dataset(metadata)

def test_VideoDataFrame_object(dataset):
    assert dataset.features.active is None
    assert dataset.metadata.n_videos == 5
    assert len(dataset.metadata.reverse_label_key) == 1

def test_df_renaming(metadata, default_track_cols):
    none_renamer = {}
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = create_dataset(metadata, part_renamer = none_renamer, animal_renamer=animal_renamer)
    df.features.active = df.pose.raw_track_columns
    assert df.features.active == default_track_cols

    new_parts = ['nose', 'left_ear', 'right_ear', 'neck', 'lefthip', 'righthip', 'tail']
    part_renamer = {'leftear': 'left_ear', 'rightear': 'right_ear'}
    df = create_dataset(metadata, part_renamer = part_renamer)
    assert set(new_parts) == set(df.pose.body_parts)

    new_animals = ['resident', 'intruder']
    animal_renamer = {'adult': 'resident', 'juvenile': 'intruder'}
    df = create_dataset(metadata, animal_renamer = animal_renamer)
    assert set(new_animals) == set(df.pose.animals)

def test_list_features():
    from ethome.features import list_inbuilt_features
    list_inbuilt_features()

def test_rf_baselinemodel(dataset):
    from ethome import add_randomforest_predictions
    from ethome.features import Distances
    distances = Distances()
    dataset.features.add(distances, 
                     featureset_name = 'distances', 
                     add_to_features = True)
    add_randomforest_predictions(dataset)
    assert 'prediction' in dataset.columns

def test_dl_features(dataset):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from ethome.features import CNN1DProb
    cnnprobs = CNN1DProb()
    dataset.features.add(cnnprobs, 
                         featureset_name = '1dcnn', 
                         add_to_features = True)
    assert set(dataset.features.active) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_dl_features_with_missing(dataset):
    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from ethome.features import CNN1DProb
    dataset.iloc[:10,0] = np.nan
    cnnprobs = CNN1DProb()
    dataset.features.add(cnnprobs, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)
    assert set(dataset.features.active) == set(['1dcnn__prob_attack', '1dcnn__prob_investigation', '1dcnn__prob_mount', '1dcnn__prob_other'])

def test_regex_features(dataset):
    dataset.features.activate('likelihood')
    features = dataset.features.regex('likelihood')
    assert len(features) == 14

def test_add_likelihood(dataset):
    new_cols = dataset.features.activate('likelihood')
    assert len(new_cols) == 14

def test_ml_accessors(dataset):
    dataset.features.activate('likelihood')
    assert dataset.ml.features.shape[1] == 14
    assert len(dataset.ml.labels) == 45749
    assert len(dataset.ml.group) == 45749

def test_save_load(tmp_path, dataset):
    from ethome.video import load_experiment
    import os
    import numpy as np
    tmp_file = os.path.join(tmp_path, 'test.pkl')
    dataset.features.activate('likelihood')
    dataset.io.save(tmp_file)
    dataset2 = load_experiment(tmp_file)
    assert dataset2.ml.features.shape == dataset.ml.features.shape
    assert np.array_equal(dataset2.ml.labels, dataset.ml.labels)
    assert np.array_equal(dataset2.ml.group, dataset.ml.group)

def test_remove_likelihood(dataset):
    new_cols = dataset.features.activate('likelihood')
    old_cols = dataset.features.deactivate('likelihood')
    assert new_cols == old_cols

def test_mars_features(dataset):
    from ethome.features import MARS
    mars_features = MARS()
    dataset.features.add(mars_features, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_mars_features_with_missing(dataset):
    import numpy as np
    from ethome.features import MARS
    # 
    #Give the df missing data:
    dataset.iloc[:10,0] = np.nan

    mars = MARS()
    dataset.features.add(mars, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_mars_then_interpolate(dataset):
    from ethome.features import MARS
    mars = MARS()
    dataset.features.add(mars, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    interpolate_lowconf_points(dataset)
    assert len(dataset.features.active) == 726

def test_singleanimal_interpolate(openfield_sample):
    interpolate_lowconf_points(openfield_sample)
    assert len(openfield_sample.columns) == 15

def test_duplicate_mars_features(dataset):
    from ethome.features import MARS
    mars = MARS()
    dataset.features.add(mars, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    dataset.features.add(mars, 
                     featureset_name = 'MARS', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 726

def test_distance_features(dataset):
    from ethome.features import Distances
    distances = Distances()
    dataset.features.add(distances, 
                     featureset_name = 'distances', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_speed_features(dataset):
    from ethome.features import Speeds
    speeds = Speeds()
    dataset.features.add(speeds, 
                     featureset_name = 'speeds', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_social_features(dataset):
    from ethome.features import Social
    social = Social()
    dataset.features.add(social, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 98

def test_marsreduced_features(dataset):
    from ethome.features import MARSReduced
    marsred = MARSReduced()
    dataset.features.add(marsred, 
                     featureset_name = 'social', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 285

def test_marsreduced_features_by_string(dataset):
    dataset.features.add('mars_reduced', 
                     featureset_name = 'marsreduced', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 285

#Test new feature creation methods... use a custom function, and use a custom class, and use a string
def test_custom_feature_func(dataset):
    def diff_cols(df, required_columns = []):
        return df.loc[:,required_columns].diff()

    dataset.features.add(diff_cols, required_columns = ['resident_x_neck', 'resident_y_neck'])
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 2

def test_custom_feature_class(dataset):

    class BodyPartDiff:
        def __init__(self, required_columns):
            self.required_columns = required_columns

        def transform(self, df, **kwargs):
            return df.loc[:,self.required_columns].diff()

    head_diff = BodyPartDiff(['resident_x_neck', 'resident_y_neck'])
    dataset.features.add(head_diff)

    assert len(dataset.features.active) == 2


def test_interpolate(dataset):
    interpolate_lowconf_points(dataset)

    assert pd.notnull(dataset).all(axis = None)

def test_rescaling_metadataparams(tracking_files, metadata_params):
    """ Test creation of VideoDataFrame object with more varied metadata key combinations"""

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_width = create_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = create_dataset(metadata_no_width)
    assert 'units' not in list(df.metadata.details.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    metadata_no_widthunits = create_metadata(tracking_files, 
                                fps = metadata_params['fps'], 
                                resolution = metadata_params['resolution'])
    df = create_dataset(metadata_no_widthunits)
    assert 'units' not in list(df.metadata.details.values())[0]

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we should
    metadata_no_labels = create_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    df = create_dataset(metadata_no_labels)
    assert list(df.metadata.details.values())[0]['units'] == 'mm'

    metadata_no_labels = create_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'],
                                units = 'cm')

    df = create_dataset(metadata_no_labels)
    assert list(df.metadata.details.values())[0]['units'] == 'cm'

    #Should have 'frame_width', 'resolution' and 'frame_width_units' to make conversion. here we shouldn't
    # Because we're just missing one value in one rec
    metadata_no_labels = create_metadata(tracking_files, 
                                frame_width = metadata_params['frame_width'], 
                                fps = metadata_params['fps'], 
                                frame_width_units = metadata_params['frame_width_units'], 
                                resolution = metadata_params['resolution'])
    del list(metadata_no_labels.values())[1]['frame_width']
    df = create_dataset(metadata_no_labels)
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
    from ethome.features import CentroidInteranimal
    cinter = CentroidInteranimal()
    dataset.features.add(cinter, 
                     featureset_name = 'com_interanimal', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 1

def test_centerofmass_interanimal_speed_features(dataset):
    from ethome.features import CentroidInteranimalSpeed
    cinterspeed = CentroidInteranimalSpeed()
    dataset.features.add(cinterspeed, 
                     featureset_name = 'com_interanimal_speed', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 1

def test_centerofmass_features(dataset):
    from ethome.features  import Centroid
    c = Centroid()
    dataset.features.add(c, 
                     featureset_name = 'com', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 4

def test_centerofmass_vel_features(dataset):
    from ethome.features  import CentroidVelocity
    cspeed = CentroidVelocity()
    dataset.features.add(cspeed, 
                     featureset_name = 'com_vel', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 4

def test_speed_features(dataset):
    from ethome.features  import Speeds
    speeds = Speeds()
    dataset.features.add(speeds, 
                     featureset_name = 'speed', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91

def test_dist_features(dataset):
    from ethome.features  import Distances
    distances = Distances()
    dataset.features.add(distances, 
                     featureset_name = 'dist', 
                     add_to_features = True)
    #Check we made the right amount of new columns
    assert len(dataset.features.active) == 91