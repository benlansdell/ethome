""" Functions to take pose tracks and compute a set of features from them.
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

class Features: # pragma: no cover
    def __init__(self):
        raise NotImplementedError

    def transform(self, df):
        raise NotImplementedError

def feature_class_maker(name, compute_function, required_columns = []):
    def __init__(self, required_columns = None, **kwargs):
        """Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns.

        See docstring for the `features` model for more information.

        Args:
            required_columns: The columns that are required to compute the features.
        """
        if required_columns is not None:
            self.required_columns = required_columns
        self.kwargs = kwargs

    def fit(self, edf, **kwargs): # pragma: no cover
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
        new_features = self.feature_maker(edf, self.required_columns, **self.kwargs, **kwargs)
        return new_features   

    def fit_transform(self, edf, **kwargs): # pragma: no cover
        self.fit(edf, **kwargs)
        return self.transform(edf, **kwargs)
    
    return type(name, (Features,), {
        '__init__': __init__,
        'fit': fit,
        'transform': transform,
        'fit_transform': fit_transform,
        'required_columns': required_columns,
        'feature_maker': staticmethod(compute_function)
    })

## Built in feature makers

#Social mice studies
MARS = feature_class_maker('MARSFeatures', compute_mars_features, default_tracking_columns)
MARSReduced = feature_class_maker('MARSReduced', compute_mars_reduced_features, default_tracking_columns)
CNN1DProb = feature_class_maker('CNN1DProb', compute_dl_probability_features, default_tracking_columns)
Social = feature_class_maker('Social', compute_social_features, default_tracking_columns)

## Generic features -- don't need any specific column names. Will be based on the animal setup.

#Distances between all animals' centroids
CentroidInteranimal = feature_class_maker('CentroidInteranimal', compute_centerofmass_interanimal_distances)

#Speeds between all animals' centroids
CentroidInteranimalSpeed = feature_class_maker('CentroidInteranimalSpeed', compute_centerofmass_interanimal_speed)

#Centroid of all animals
Centroid = feature_class_maker('Centroid', compute_centerofmass)

#Velocity of all animals' centroids
CentroidVelocity = feature_class_maker('CentroidVelocity', compute_centerofmass_velocity)

#Speeds between all body parts pairs (within and between animals)
Speeds = feature_class_maker('Speeds', compute_speed_features)

#Distances between all body parts pairs (within and between animals)
Distances = feature_class_maker('Distances', compute_distance_features)

FEATURE_MAKERS = {'mars': MARS,
                  'mars_reduced': MARSReduced,
                  'cnn1d_prob': CNN1DProb,
                  'social': Social,
                  'centroids_interanimal': CentroidInteranimal,
                  'centroids_interanimal_speed': CentroidInteranimalSpeed,
                  'centroids': Centroid,
                  'centroids_velocity': CentroidVelocity,
                  'intrabodypartspeeds': Speeds,
                  'intrabodypartdistances': Distances,
                  'distances': Distances}

FEATURE_MAKERS_DESCRIPTION = [
    ['MARS', 'mars', 'MARS mouse resident-intruder features'],
    ['MARS Reduced', 'mars_reduced', 'MARS mouse resident-intruder features, reduced to 10 features'],
    ['CNN1D Prob', 'cnn1d_prob', 'CNN1D probability features'],
    ['Social', 'social', 'Social features'],
    ['Centroids Interanimal', 'centroids_interanimal', 'Centroids interanimal features'],
    ['Centroids Interanimal Speed', 'centroids_interanimal_speed', 'Centroids interanimal speed features'],
    ['Centroids', 'centroids', 'Centroids features'],
    ['Centroids Velocity', 'centroids_velocity', 'Centroids velocity features'],
    ['Intrabodypart Speeds', 'intrabodypartspeeds', 'Intrabodypart speeds features'],
    ['Intrabodypart Distances', 'intrabodypartdistances', 'Intrabodypart distances features']
]

def list_inbuilt_features():
    """Print available feature makers
    
    Args:
    
    Returns:
        None. Prints output
    """
    print("Available feature makers: (long name, short name, description)")
    print("Use shortname, e.g. df.features.add(<short name>)")
    for long_name, short_name, desc in FEATURE_MAKERS_DESCRIPTION:
        print(f'{long_name}, {short_name}, {desc}')
