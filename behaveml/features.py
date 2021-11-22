""" Functions to take pose tracks and compute a set of features from them """
import pandas as pd

from typing import Callable

from behaveml.dl.dl_features import compute_dl_probability_features
from behaveml.mars_features import compute_mars_features

default_tracking_columns = ['adult_x_nose', 'adult_x_leftear', 'adult_x_rightear', 'adult_x_neck',
                            'adult_x_lefthip', 'adult_x_righthip', 'adult_x_tail', 'adult_y_nose',
                            'adult_y_leftear', 'adult_y_rightear', 'adult_y_neck',
                            'adult_y_lefthip', 'adult_y_righthip', 'adult_y_tail',
                            'juvenile_x_nose', 'juvenile_x_leftear', 'juvenile_x_rightear',
                            'juvenile_x_neck', 'juvenile_x_lefthip', 'juvenile_x_righthip',
                            'juvenile_x_tail', 'juvenile_y_nose', 'juvenile_y_leftear',
                            'juvenile_y_rightear', 'juvenile_y_neck', 'juvenile_y_lefthip',
                            'juvenile_y_righthip', 'juvenile_y_tail']

class Features(object):
    def __init__(self, feature_maker : Callable, required_columns : list, **kwargs):
        self.required_columns = required_columns
        self.feature_maker = feature_maker

    def make(self, vdf, **kwargs):
        #Validate columns:
        checks = [col in self.required_columns for col in vdf.data.columns]
        if sum(checks) < len(self.required_columns):
            raise RuntimeError("VideoDataFrame doesn't have necessary columns to compute this set of features.")
        new_features = self.feature_maker(vdf.data, self.required_columns, vdf.animal_setup, **kwargs)
        return new_features

mars_feature_maker = Features(compute_mars_features, default_tracking_columns)
cnn_probability_feature_maker = Features(compute_dl_probability_features, default_tracking_columns)