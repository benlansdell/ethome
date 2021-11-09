#######################
## General todo list ##
#######################

#TODO
# * Import MARS feature creation 

###########################
## Example analysis code ##
###########################

from glob import glob 
from behaveml import VideosetDataFrame, clone_metadata
from behaveml import compute_dl_probability_features, create_mars_features

#from behaveml import read_DLC_tracks

#A list of DLC tracking files
tracking_files = glob('./tests/data/dlc/*.csv')

#A list of BORIS labeled files, ordered to correspond with DLC tracking file list
boris_files = glob('./tests/data/boris/*.csv')

frame_length = None              # (float) length of entire horizontal shot
units = None                     # (str) units frame_length is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels

#Metadata is a dictionary
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          frame_length = frame_length, 
                          fps = fps, 
                          units = units, 
                          resolution = resolution)

dataset = VideosetDataFrame(metadata)

#Now create features on this dataset
dataset.add_features(compute_dl_probability_features, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)

####################
# Works up to here # 
####################

dataset.add_features(create_mars_features, 
                     featureset_name = 'MARS', 
                     add_to_features = True)

#You can make your own 'feature maker':
#The feature creation functions take:
# * Pandas data frame (dataset.data)
# * A list of column names that indicate which columns in the data frame 
#   the features will be computed from. For basic feature, this will be the 'raw' pose-tracking column names
#   For 'model stacking' models, this can be derivative column sets
# * A dictionary, animal_setup, that contains details about the animals in the experiment
# They return 
# * Pandas data frame with the new features, in the same order as input data

#dataset.add_features(create_custom_features, 
#                     featureset_name = 'custom', 
#                     add_to_feature_col = True)

#Now all the work of the package is done, more or less, and 
# we can do ML on this object in e.g. sklearn by using the following attributes:
# dataset.features #The feature matrix setup for learning w
# dataset.label    #The labels for supervision
# dataset.group    #Used for group-level cross validation 
#                   (by default, groups are set to filename, so this implements video-level CV)