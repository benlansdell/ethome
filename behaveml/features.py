""" Functions to take pose tracks and compute a set of features from them """

from typing import Callable

from behaveml.dl.dl_features import compute_dl_probability_features
from behaveml.mars_features import compute_mars_features, compute_distance_features, compute_velocity_features, \
                                   compute_mars_reduced_features, compute_social_features

# default_tracking_columns = ['adult_x_nose', 'adult_x_leftear', 'adult_x_rightear', 'adult_x_neck',
#                             'adult_x_lefthip', 'adult_x_righthip', 'adult_x_tail', 'adult_y_nose',
#                             'adult_y_leftear', 'adult_y_rightear', 'adult_y_neck',
#                             'adult_y_lefthip', 'adult_y_righthip', 'adult_y_tail',
#                             'juvenile_x_nose', 'juvenile_x_leftear', 'juvenile_x_rightear',
#                             'juvenile_x_neck', 'juvenile_x_lefthip', 'juvenile_x_righthip',
#                             'juvenile_x_tail', 'juvenile_y_nose', 'juvenile_y_leftear',
#                             'juvenile_y_rightear', 'juvenile_y_neck', 'juvenile_y_lefthip',
#                             'juvenile_y_righthip', 'juvenile_y_tail']

default_tracking_columns = ['resident_x_nose', 'resident_x_leftear', 'resident_x_rightear', 'resident_x_neck',
                            'resident_x_lefthip', 'resident_x_righthip', 'resident_x_tail', 'resident_y_nose',
                            'resident_y_leftear', 'resident_y_rightear', 'resident_y_neck',
                            'resident_y_lefthip', 'resident_y_righthip', 'resident_y_tail',
                            'intruder_x_nose', 'intruder_x_leftear', 'intruder_x_rightear',
                            'intruder_x_neck', 'intruder_x_lefthip', 'intruder_x_righthip',
                            'intruder_x_tail', 'intruder_y_nose', 'intruder_y_leftear',
                            'intruder_y_rightear', 'intruder_y_neck', 'intruder_y_lefthip',
                            'intruder_y_righthip', 'intruder_y_tail']

class Features(object):
    def __init__(self, feature_maker : Callable, required_columns : list, **kwargs):
        self.required_columns = required_columns
        self.feature_maker = feature_maker

    def make(self, vdf, **kwargs):
        #Validate columns:
        checks = [col in self.required_columns for col in vdf.data.columns]
        if sum(checks) < len(self.required_columns):
            raise RuntimeError("VideosetDataFrame doesn't have necessary columns to compute this set of features.")
        new_features = self.feature_maker(vdf.data, self.required_columns, vdf.animal_setup, **kwargs)
        return new_features

mars_feature_maker = Features(compute_mars_features, default_tracking_columns)
cnn_probability_feature_maker = Features(compute_dl_probability_features, default_tracking_columns)
distance_feature_maker = Features(compute_distance_features, default_tracking_columns)
marsreduced_feature_maker = Features(compute_mars_reduced_features, default_tracking_columns)
social_feature_maker = Features(compute_social_features, default_tracking_columns)
velocity_feature_maker = Features(compute_velocity_features, default_tracking_columns)

#To make your own:
# Create a function 'create_custom_features' and provide the Features class a list of columns
# that are needed by this function to compute the features.
# 'create_custom_features' should take:
# compute_social_features(<df>, <raw_col_names>, <animal_setup>, **kwargs)
# where:
# 'df' is the dataframe to compute the features on
# 'raw_col_names' is a list of the names of the columns in the dataframe that contain the raw data used for the feature creation. These are required for the model.
# 'animal_setup' is a dictionary with keys 'bodypart_ids', 'mouse_ids', 'colnames'.
#    'bodypart_ids' is a list of the bodypart ids that are used in the dataframe
#    'mouse_ids' is a list of the mouse ids that are used in the dataframe
#    'colnames' is the list product(animals, XY_IDS, body_parts) 
# **kwargs are extra arguments passed onto the feature creation function.
# The function returns:
# A dataframe with the new features. These will be added to the VideosetDataFrame as columns.

