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

A dataframe, that only contains the new features. These will be added to the ExperimentDataFrame as columns.

Once you have such a function defined, you can create a "feature making object" with

`custom_feature_maker = Features(create_custom_features, req_columns)`

This could be used on datasets as:

```
dataset.add_features(custom_feature_maker, featureset_name = 'CUSTOM', add_to_features = True)
```
"""

import warnings

from ethome.features.dl_features import compute_dl_probability_features
from ethome.features.mars_features import compute_mars_features, compute_mars_reduced_features, compute_social_features
from ethome.features.generic_features import compute_centerofmass_interanimal_distances, \
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

def feature_class_maker(name, compute_function, required_columns):
    def __init__(self, required_columns = None, **kwargs):
        """Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns.

        See docstring for the `features` model for more information.

        Args:
            required_columns: The columns that are required to compute the features.
        """
        if required_columns is not None:
            self.required_columns = required_columns
        self.kwargs = kwargs

    def fit(self, edf, **kwargs):
        return 

    def transform(self, edf, **kwargs):
        """Make the features. This is called internally by the dataset object when running `add_features`.

        Args:
            edf: The ExperimentDataFrame to compute the features on.
            **kwargs: Extra arguments passed onto the feature creation function.
        """
        #Validate columns:
        checks = [col in self.required_columns for col in edf.columns]
        if sum(checks) < len(self.required_columns):
            raise RuntimeError("DataFrame doesn't have necessary columns to compute this set of features.")
        if edf[self.required_columns].isnull().values.any():
            warnings.warn("Missing values in required data columns. May result in unexpected behavior. Consider interpolating or imputing missing data first.")
        new_features = self.feature_maker(edf, self.required_columns, edf.pose.animal_setup, **self.kwargs, **kwargs)
        return new_features   

    def fit_transform(self, edf, **kwargs):
        self.fit(edf, **kwargs)
        return self.transform(edf, **kwargs)
    
    return type(name, (object,), {
        '__init__': __init__,
        'fit': fit,
        'transform': transform,
        'fit_transform': fit_transform,
        'required_columns': required_columns,
        'feature_maker': staticmethod(compute_function)
    })

## Built in feature makers

MARS = feature_class_maker('MARSFeatures', compute_mars_features, default_tracking_columns)
MARSReduced = feature_class_maker('MARSReduced', compute_mars_reduced_features, default_tracking_columns)
CNN1DProb = feature_class_maker('CNN1DProb', compute_dl_probability_features, default_tracking_columns)
Social = feature_class_maker('Social', compute_social_features, default_tracking_columns)

## Generic features -- don't need any specific column names. Will be based on the animal setup.
CentroidInteranimal = feature_class_maker('CentroidInteranimal', compute_centerofmass_interanimal_distances, [])
CentroidInteranimalSpeed = feature_class_maker('CentroidInteranimalSpeed', compute_centerofmass_interanimal_speed, [])
Centroid = feature_class_maker('Centroid', compute_centerofmass, [])
CentroidVelocity = feature_class_maker('CentroidVelocity', compute_centerofmass_velocity, [])
Speeds = feature_class_maker('Speeds', compute_speed_features, [])
Distances = feature_class_maker('Distances', compute_distance_features, [])