""" Functions to take pose tracks and compute a set of features from them.

To make your own feature creator:

Create a function, e.g. `create_custom_features`, and provide the Features class a list of columns that are needed by this function to compute the features.

The function `create_custom_features` has the form:

`create_custom_features(<df>, <raw_col_names>, <animal_setup>, **kwargs)`

Where:

`df` is the dataframe to compute the features on
`raw_col_names` is a list of the names of the columns in the dataframe that contain the raw data used for the feature creation. These are required for the model.
`animal_setup` is a dictionary with keys `bodypart_ids`, `mouse_ids`, `colnames`.
   `bodypart_ids` is a list of the bodypart ids that are used in the dataframe
   `mouse_ids` is a list of the mouse ids that are used in the dataframe
   `colnames` is the list product(animals, XY_IDS, body_parts) 
`**kwargs` are extra arguments passed onto the feature creation function.

The function returns:

A dataframe, that only contains the new features. These will be added to the VideosetDataFrame as columns.

Once you have such a function defined, you can create a "feature making object" with

`custom_feature_maker = Features(create_custom_features, req_columns)`

This could be used on datasets as:

```
dataset.add_features(custom_feature_maker, featureset_name = 'CUSTOM', add_to_features = True)
```
"""

from typing import Callable
import warnings

from behaveml.dl.dl_features import compute_dl_probability_features
from behaveml.mars_features import compute_mars_features, compute_mars_reduced_features, compute_social_features
from behaveml.generic_features import compute_centerofmass_interanimal_distances, \
                                        compute_centerofmass_interanimal_speed, \
                                        compute_centerofmass, \
                                        compute_centerofmass_velocity, \
                                        compute_speed_features, \
                                        compute_distance_features

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
        """Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns.

        See docstring for the `features` model for more information.

        Args:
            feature_maker: The function that will be used to compute the features.
            required_columns: The columns that are required to compute the features.
        """
        self.required_columns = required_columns
        self.feature_maker = feature_maker
        self.kwargs = kwargs

    def make(self, vdf, **kwargs):
        """Make the features. This is called internally by the dataset object when running `add_features`.

        Args:
            vdf: The VideosetDataFrame to compute the features on.
            **kwargs: Extra arguments passed onto the feature creation function.
        """
        #Validate columns:
        checks = [col in self.required_columns for col in vdf.data.columns]
        if sum(checks) < len(self.required_columns):
            raise RuntimeError("VideosetDataFrame doesn't have necessary columns to compute this set of features.")
        if vdf.data[self.required_columns].isnull().values.any():
            warnings.warn("Missing values in required data columns. May result in unexpected behavior. Consider interpolating or imputing missing data first.")
        new_features = self.feature_maker(vdf.data, self.required_columns, vdf.animal_setup, **self.kwargs, **kwargs)
        return new_features

## MARS features
mars_feature_maker = Features(compute_mars_features, default_tracking_columns)
marsreduced_feature_maker = Features(compute_mars_reduced_features, default_tracking_columns)
cnn_probability_feature_maker = Features(compute_dl_probability_features, default_tracking_columns)
social_feature_maker = Features(compute_social_features, default_tracking_columns)

## Generic features -- don't need any specific column names. Will be based on the animal setup.
com_interanimal_feature_maker = Features(compute_centerofmass_interanimal_distances, [])
com_interanimal_speed_feature_maker = Features(compute_centerofmass_interanimal_speed, [])
com_feature_maker = Features(compute_centerofmass, [])
com_velocity_feature_maker = Features(compute_centerofmass_velocity, [])
speed_feature_maker = Features(compute_speed_features, [])
distance_feature_maker = Features(compute_distance_features, [])