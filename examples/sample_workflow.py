###########################
## Example analysis code ##
###########################

from glob import glob 
from behaveml import VideosetDataFrame, clone_metadata

#A list of DLC tracking files
tracking_files = glob('./testdata/dlc/*.csv')

#A list of BORIS labeled files, ordered to correspond with DLC tracking file list
boris_files = glob('./testdata/boris/*.csv')

shot_length = None              # (float) length of entire horizontal shot
units = None                    # (str) units show_length is given in
fps = 30                        # (int) frames per second
resolution = (1200, 1600)       # (tuple) HxW in pixels

#Metadata is a dictionary
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          shot_length = shot_length, 
                          fps = fps, 
                          units = units, 
                          resolution = resolution)

dataset = VideosetDataFrame(metadata)

#Now create features on this dataset
dataset.create_dl_features()
dataset.create_mabe_features()
dataset.create_custom_features()
dataset.add_features()

#Set features by group names

#Now we can do ML on this object with the following attributes:
# dataset.features
# dataset.label
# dataset.splitter and/or 
# dataset.group