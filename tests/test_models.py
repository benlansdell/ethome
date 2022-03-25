#####################
## Setup test code ##
#####################

import pytest
from behaveml import VideosetDataFrame, clone_metadata, interpolate_lowconf_points, video
import pandas as pd

#Metadata is a dictionary
# def test_f1_optimizer(videodataset_mars):
#     """ Test creation of metadata object """

#     model = F1Optimizer()
#     model.fit()